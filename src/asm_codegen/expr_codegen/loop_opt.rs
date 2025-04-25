use std::collections::{HashMap, HashSet, VecDeque};

use crate::{
    asm_codegen::{Asm, AsmEnv, Instr, MemLoc, Operand, Reg, WORD_SIZE},
    ast::{
        auxiliary::LoopVar,
        expr::{Expr, ExprKind},
    },
    typecheck::TypeVal,
};

#[derive(Debug, Clone)]
pub struct TensorInfo<'a> {
    pub topo_order: Vec<&'a str>,
    pub array: (&'a [LoopVar], &'a Expr, usize),
    pub sum: (&'a [LoopVar], &'a Expr, usize),
}

impl<'a> AsmEnv<'a> {
    pub fn is_tensor(&self, expr: &'a Expr) -> Option<TensorInfo<'a>> {
        let (arr_loop_vars, arr_body, arr_scope) =
            if let ExprKind::ArrayComp(loop_vars, body, scope) = expr.kind() {
                (loop_vars, body, scope)
            } else {
                return None;
            };

        let (sum_loop_vars, sum_body, sum_scope) =
            if let ExprKind::Sum(loop_vars, body, scope) = arr_body.kind() {
                (loop_vars, body, scope)
            } else {
                return None;
            };

        let vertices: Vec<_> = arr_loop_vars
            .iter()
            .chain(sum_loop_vars)
            .map(|LoopVar(n, _)| n.as_str(self.env.src()))
            .collect();

        let mut edges = Vec::new();
        if !self.is_tensor_body(&sum_body, &vertices, &mut edges) {
            return None;
        }

        let mut adjacency_list: HashMap<&str, HashSet<&str>> = HashMap::new();
        for (src, dest) in &edges {
            adjacency_list.entry(src).or_default().insert(dest);
        }

        let mut topo_order = Vec::new();

        let mut in_edges: HashMap<&'a str, usize> =
            HashMap::from_iter(vertices.iter().map(|s| (*s, 0)));
        for (_, target) in &edges {
            *in_edges.get_mut(target).unwrap() += 1;
        }

        let mut q = VecDeque::from_iter(in_edges.iter().filter(|(_, v)| **v == 0).map(|(k, _)| *k));

        while let Some(cur) = q.pop_front() {
            topo_order.push(cur);
            for target in adjacency_list.get(cur).unwrap_or(&HashSet::new()) {
                *in_edges.get_mut(target).unwrap() -= 1;
                if in_edges[target] == 0 {
                    q.push_back(target);
                }
            }
        }

        if topo_order.len() == vertices.len() {
            Some(TensorInfo {
                topo_order,
                array: (arr_loop_vars, arr_body, *arr_scope),
                sum: (sum_loop_vars, sum_body, *sum_scope),
            })
        } else {
            None
        }
    }

    fn is_tensor_body(
        &self,
        expr: &Expr,
        vertices: &[&str],
        edges: &mut Vec<(&'a str, &'a str)>,
    ) -> bool {
        (match expr.kind() {
            ExprKind::ArrayIndex(expr, indices) => {
                if indices.iter().all(|i| Self::is_tensor_primitive(i))
                    && Self::is_tensor_primitive(expr)
                {
                    let indices: Vec<_> = indices
                        .iter()
                        .filter(|i| matches!(i.kind(), ExprKind::Var))
                        .map(|v| v.loc().as_str(self.env.src()))
                        .filter(|v| vertices.contains(v))
                        .collect();

                    edges.extend(indices.windows(2).map(|w| (w[0], w[1])));
                    true
                } else {
                    false
                }
            }
            ExprKind::And(args)
            | ExprKind::Or(args)
            | ExprKind::LessThan(args)
            | ExprKind::GreaterThan(args)
            | ExprKind::LessThanEq(args)
            | ExprKind::GreaterThanEq(args)
            | ExprKind::Eq(args)
            | ExprKind::NotEq(args)
            | ExprKind::Add(args)
            | ExprKind::Minus(args)
            | ExprKind::Mulitply(args)
            | ExprKind::Divide(args)
            | ExprKind::Modulo(args) => args
                .iter()
                .all(|arg| self.is_tensor_body(arg, vertices, edges)),

            ExprKind::Paren(expr) => self.is_tensor_body(expr, vertices, edges),

            _ => false,
        }) || AsmEnv::is_tensor_primitive(expr)
    }

    fn is_tensor_primitive(expr: &Expr) -> bool {
        matches!(
            expr.kind(),
            ExprKind::Var | ExprKind::IntLit(_) | ExprKind::FloatLit(_)
        )
    }

    pub fn codegen_tensor(
        &mut self,
        TensorInfo {
            topo_order,
            array: (arr_looping_vars, arr_body, _),
            sum: (sum_looping_vars, sum_body, sum_scope),
        }: TensorInfo,
    ) {
        self.add_asm([Asm::Comment("TENSOR DETECTED")]);
        let element_size = self.env.type_size(arr_body.type_data());
        self.add_instrs([Instr::Sub(
            Operand::Reg(Reg::Rsp),
            Operand::Value(WORD_SIZE),
        )]);
        let arr_rank = arr_looping_vars.len();
        let sum_rank = sum_looping_vars.len();

        let loop_rank = arr_rank + sum_rank;
        self.check_loop_bounds(arr_looping_vars);
        self.check_loop_bounds(sum_looping_vars);

        self.alloc_array(
            arr_looping_vars,
            element_size,
            sum_rank as i64 * WORD_SIZE as i64,
        );
        self.store_loop_data(WORD_SIZE as i64 * loop_rank as i64);
        self.init_loop_vars(arr_looping_vars);
        self.init_loop_vars(sum_looping_vars);
        let loop_begining = self.gen_loop_body(sum_body, sum_scope);

        self.calculate_array_index(
            arr_rank as u64,
            arr_looping_vars.iter().map(|LoopVar(_, expr)| expr),
            element_size,
            element_size + sum_rank as u64 * WORD_SIZE,
            sum_rank as u64 * WORD_SIZE,
        );

        match sum_body.type_data() {
            TypeVal::Float => {
                self.add_instrs([
                    Instr::Pop(Reg::Xmm0),
                    Instr::Add(Operand::Reg(Reg::Xmm0), Operand::Mem(MemLoc::Reg(Reg::Rax))),
                    Instr::Mov(Operand::Mem(MemLoc::Reg(Reg::Rax)), Operand::Reg(Reg::Xmm0)),
                ]);
            }
            TypeVal::Int => {
                todo!()
            }
            _ => unreachable!(),
        }

        let mut offsets: Vec<_> = sum_looping_vars
            .iter()
            .chain(arr_looping_vars)
            .enumerate()
            .map(|(i, LoopVar(name_loc, _))| {
                let index_offset = i as i64 * WORD_SIZE as i64;
                (
                    (
                        index_offset,
                        loop_rank as i64 * WORD_SIZE as i64 + index_offset,
                    ),
                    name_loc.as_str(self.env.src()),
                )
            })
            .collect();
        offsets.sort_by_key(|(_, name)| topo_order.iter().position(|a| a == name).unwrap());

        self.gen_loop_increment(offsets.into_iter().map(|(a, _)| a), loop_begining);

        self.add_instrs([Instr::Add(
            Operand::Reg(Reg::Rsp),
            Operand::Value(loop_rank as u64 * WORD_SIZE),
        )]);

        self.add_instrs([Instr::Add(
            Operand::Reg(Reg::Rsp),
            Operand::Value(sum_rank as u64 * WORD_SIZE),
        )]);
    }
}
