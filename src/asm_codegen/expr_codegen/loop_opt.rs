use std::collections::{HashMap, HashSet, VecDeque};

use crate::{
    asm_codegen::AsmEnv,
    ast::{
        auxiliary::LoopVar,
        expr::{Expr, ExprKind},
    },
};

impl<'a> AsmEnv<'a> {
    pub fn is_tensor(&self, expr: &Expr) -> Option<Vec<&'a str>> {
        let (arr_loop_vars, arr_body, arr_scope) =
            if let ExprKind::ArrayComp(loop_vars, body, scope) = expr.kind() {
                (loop_vars, body, scope)
            } else {
                return None;
            };

        //TODO: check for all contants in loop_vars

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
            let x = adjacency_list.entry(src).or_default().insert(dest);
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
            for target in &adjacency_list[cur] {
                *in_edges.get_mut(target).unwrap() -= 1;
                if in_edges[target] == 0 {
                    q.push_back(target);
                }
            }
        }

        if topo_order.len() == vertices.len() {
            Some(topo_order)
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
        match expr.kind() {
            ExprKind::ArrayIndex(expr, indices) => {
                if indices.iter().all(|i| Self::is_tensor_primitive(i)) {
                    false
                } else {
                    let indices: Vec<_> = indices
                        .iter()
                        .filter(|i| matches!(i.kind(), ExprKind::Var))
                        .map(|v| v.loc().as_str(self.env.src()))
                        .filter(|v| vertices.contains(v))
                        .collect();

                    edges.extend(indices.windows(2).map(|w| (w[0], w[1])));
                    true
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
        }
    }

    fn is_tensor_primitive(expr: &Expr) -> bool {
        match expr.kind() {
            ExprKind::Var | ExprKind::IntLit(_) | ExprKind::FloatLit(_) => true,
            _ => false,
        }
    }
}
