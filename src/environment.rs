use crate::{
    asm_codegen::{FLOAT_REGS_FOR_ARGS, INT_REGS_FOR_ARGS, WORD_SIZE},
    ast::{
        auxiliary::{Binding, LValue, LoopVar, StructField, Var},
        types::Type,
    },
    typecheck::TypeVal,
    utils::Span,
};
use builtins::{builtin_fns, builtin_structs, builtin_vars};
use miette::{miette, LabeledSpan, Severity};
use std::collections::{hash_map::Entry, HashMap};

pub mod builtins;

#[derive(Debug, Clone)]
pub struct Environment<'a> {
    src: &'a [u8],
    struct_ids: HashMap<&'a str, usize>,
    // Indexed by the struct id
    struct_info: Vec<StructInfo<'a>>,
    functions: HashMap<&'a str, FunctionInfo<'a>>,
    cur_scope: usize,
    scopes: Vec<Scope<'a>>,
}

pub const GLOBAL_SCOPE_ID: usize = 0;
const LOCAL_STACK_OFFSET: i64 = 8;
const GLOBAL_STACK_OFFSET: i64 = 16;

#[derive(Debug, Clone)]
pub struct StructInfo<'a> {
    fields: Box<[(&'a str, TypeVal)]>,
    id: usize,
    name: &'a str,
    size: u64,
}

#[derive(Debug, Clone)]
pub struct FunctionInfo<'a> {
    args: Box<[TypeVal]>,
    ret: TypeVal,
    name: &'a str,
    scope: usize,
}

impl FunctionInfo<'_> {
    pub fn args(&self) -> &[TypeVal] {
        &self.args
    }

    pub fn ret(&self) -> &TypeVal {
        &self.ret
    }

    pub fn name(&self) -> &str {
        self.name
    }
}

/// Represents a all the variables in a scope in JPL.
#[derive(Debug, Clone)]
pub struct Scope<'a> {
    names: HashMap<&'a str, VarInfo>,
    // Stores with size of this scope plus all parent scopes until we hit a function scope
    cur_size: u64,
    /// The index of its parent scope in the vec of scopes. 0 is always the index of the global
    /// scope. The parent of the global scope will be itself
    parent: usize,
    in_fn: bool,
}

#[derive(Debug, Clone)]
pub struct VarInfo {
    var_type: TypeVal,
    stack_loc: StackLoc,
    //bindings: Box<[&'a str]>,
}

#[derive(Debug, Clone, Copy)]
enum StackLoc {
    Local(i64),
    Global(i64),
}
impl StackLoc {
    fn offset(&mut self, offset: i64) {
        match self {
            StackLoc::Local(cur) | StackLoc::Global(cur) => *cur += offset,
        }
    }
}

impl StructInfo<'_> {
    pub fn fields(&self) -> &[(&str, TypeVal)] {
        &self.fields
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn name(&self) -> &str {
        self.name
    }
}

impl<'a> Environment<'a> {
    pub fn new(src: &'a [u8]) -> Self {
        let (struct_ids, struct_info) = builtin_structs();
        let functions = builtin_fns();

        Self {
            struct_ids,
            struct_info,
            src,
            functions,
            scopes: vec![builtin_vars()],
            cur_scope: GLOBAL_SCOPE_ID,
        }
    }

    pub fn add_struct(&mut self, Var(name): Var, params: &[StructField]) -> miette::Result<()> {
        let name_str = name.as_str(self.src);
        if self.struct_ids.contains_key(name_str) {
            return Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::new(
                    Some(format!("Struct {} already declared", name_str)),
                    name.start(),
                    name.len(),
                )],
                "Redefinition of struct"
            ));
        }

        let mut fields = Vec::with_capacity(params.len());
        let mut field_names = HashMap::with_capacity(params.len());

        for StructField(Var(field_name), field_type) in params {
            let field_name_str = field_name.as_str(self.src);
            let entry = match field_names.entry(field_name_str) {
                Entry::Occupied(entry) => {
                    let dup_loc: &Span = entry.get();
                    return Err(miette!(
                        severity = Severity::Error,
                        labels = vec![
                            LabeledSpan::new(
                                Some("first delclared here".to_string()),
                                dup_loc.start(),
                                dup_loc.len(),
                            ),
                            LabeledSpan::new(
                                Some("then here".to_string()),
                                field_name.start(),
                                field_name.len(),
                            ),
                        ],
                        "Duplicate field: {} name for Struct: {}",
                        field_name_str,
                        name_str
                    ));
                }
                Entry::Vacant(vacant_entry) => vacant_entry,
            };

            let converted_type = TypeVal::from_ast_type(field_type, self)?;
            entry.insert(*field_name);
            fields.push((field_name_str, converted_type))
        }
        let size = fields.iter().map(|(_, f)| self.type_size(f)).sum();

        let id = self.struct_info.len();
        self.struct_ids.insert(name_str, id);

        self.struct_info.push(StructInfo {
            fields: fields.into_boxed_slice(),
            id,
            name: name_str,
            size,
        });

        Ok(())
    }

    pub fn get_struct(&self, loc: Span) -> miette::Result<&StructInfo> {
        let id = self
            .struct_ids
            .get(loc.as_str(self.src))
            .copied()
            .ok_or_else(|| {
                let struct_name = loc.as_str(self.src);
                miette!(
                    severity = Severity::Error,
                    labels = vec![LabeledSpan::new(
                        Some(format!("could not find struct: {}", struct_name)),
                        loc.start(),
                        loc.len(),
                    )],
                    "Unkonwn struct: {}",
                    struct_name,
                )
            })?;
        Ok(&self.struct_info[id])
    }

    pub fn get_struct_id(&self, id: usize) -> &StructInfo {
        &self.struct_info[id]
    }

    pub fn src(&self) -> &'a [u8] {
        self.src
    }

    pub fn struct_info(&self) -> &[StructInfo<'a>] {
        &self.struct_info
    }

    pub fn add_function(
        &mut self,
        name: Span,
        args: &[Binding],
        ret_type: &Type,
    ) -> miette::Result<usize> {
        assert!(self.cur_scope == GLOBAL_SCOPE_ID);
        let scope = self.new_scope_with_size(0, true);
        let ret = TypeVal::from_ast_type(ret_type, self)?;

        self.check_name_free(name)?;
        let name = name.as_str(self.src);
        let mut g_regs_left = INT_REGS_FOR_ARGS.len()
            - if matches!(ret, TypeVal::Array(_, _) | TypeVal::Struct(_)) {
                // Add the size of the pointer to where the return type should be placed
                self.scopes[self.cur_scope].cur_size += WORD_SIZE;
                1
            } else {
                0
            };
        let mut fp_regs_left = FLOAT_REGS_FOR_ARGS.len();
        let mut stack_args_offset = StackLoc::Local(-16);

        let args = args
            .iter()
            .map(|arg| {
                let arg_type = TypeVal::from_ast_type(arg.var_type(), self)?;
                let type_size = self.type_size(&arg_type);
                let ret = arg_type.clone();
                match &arg_type {
                    TypeVal::Int | TypeVal::Bool | TypeVal::Void if g_regs_left > 0 => {
                        self.add_lvalue(arg.lvalue(), arg_type, None);
                        self.scopes[self.cur_scope].cur_size += WORD_SIZE;
                        g_regs_left -= 1;
                    }
                    TypeVal::Float if fp_regs_left > 0 => {
                        self.add_lvalue(arg.lvalue(), arg_type, None);
                        self.scopes[self.cur_scope].cur_size += WORD_SIZE;
                        fp_regs_left -= 1;
                    }
                    _ => {
                        self.add_lvalue(arg.lvalue(), arg_type, Some(stack_args_offset));
                        stack_args_offset.offset(-(type_size as i64));
                    }
                }
                Ok(ret)
            })
            .collect::<miette::Result<Vec<_>>>()?
            .into_boxed_slice();

        self.functions.insert(
            name,
            FunctionInfo {
                args,
                ret,
                name,
                scope,
            },
        );

        Ok(scope)
    }

    pub fn end_scope(&mut self) {
        self.cur_scope = self.scopes[self.cur_scope].parent;
    }

    pub fn get_function(&self, name: Span) -> miette::Result<&FunctionInfo> {
        let name_str = name.as_str(self.src);
        self.functions.get(name_str).ok_or_else(|| {
            miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::new(
                    Some(format!("could not find function: {}", name_str)),
                    name.start(),
                    name.len()
                )],
                "Unkown function called"
            )
        })
    }

    fn new_scope_with_size(&mut self, size: u64, in_fn: bool) -> usize {
        self.scopes.push(Scope {
            names: HashMap::new(),
            parent: self.cur_scope,
            cur_size: size,
            in_fn,
        });

        self.cur_scope = self.scopes.len() - 1;
        self.cur_scope
    }

    pub fn new_scope(&mut self) -> usize {
        self.new_scope_with_size(
            self.scopes[self.cur_scope].cur_size,
            self.scopes[self.cur_scope].in_fn,
        )
    }

    pub fn add_loop_bounds(&mut self, loop_vars: &[LoopVar]) -> miette::Result<()> {
        // Add the space used for accumulator/pointer and looping bounds
        let cur_scope = &mut self.scopes[self.cur_scope];
        cur_scope.cur_size += WORD_SIZE as u64 + loop_vars.len() as u64 * WORD_SIZE as u64;
        for LoopVar(name, _) in loop_vars {
            self.add_name(*name, TypeVal::Int, None)?;
            self.scopes[self.cur_scope].cur_size += WORD_SIZE;
        }

        Ok(())
    }

    pub fn add_let_lvalue(&mut self, l_val: &LValue, var_type: TypeVal) -> miette::Result<()> {
        let type_size = self.type_size(&var_type);
        self.add_lvalue(l_val, var_type, None)?;
        self.scopes[self.cur_scope].cur_size += type_size;
        Ok(())
    }

    fn add_lvalue(
        &mut self,
        l_val: &LValue,
        var_type: TypeVal,
        stack_loc: Option<StackLoc>,
    ) -> miette::Result<()> {
        let mut stack_loc = self.add_name(l_val.variable().loc(), var_type, stack_loc)?;
        for Var(binding_name) in l_val.array_bindings().into_iter().flatten() {
            self.add_name(*binding_name, TypeVal::Int, Some(stack_loc));
            stack_loc.offset(8);
        }
        Ok(())
    }

    fn add_name(
        &mut self,
        name: Span,
        var_type: TypeVal,
        stack_loc: Option<StackLoc>,
    ) -> miette::Result<StackLoc> {
        self.check_name_free(name)?;
        let cur_scope = &mut self.scopes[self.cur_scope];
        let stack_loc = stack_loc.unwrap_or(if cur_scope.in_fn {
            StackLoc::Local(cur_scope.cur_size as i64 + LOCAL_STACK_OFFSET)
        } else {
            StackLoc::Global(cur_scope.cur_size as i64 + GLOBAL_STACK_OFFSET)
        });
        let name_str = name.as_str(self.src);
        cur_scope.names.insert(
            name_str,
            VarInfo {
                var_type: var_type.clone(),
                stack_loc,
            },
        );
        Ok(stack_loc)
    }

    fn check_name_free(&mut self, name: Span) -> miette::Result<()> {
        let mut current_scope_id = self.cur_scope;
        let name_str = name.as_str(self.src);
        let dup_found = loop {
            let current_scope = &self.scopes[current_scope_id];
            if current_scope.names.contains_key(name_str) {
                break true;
            }

            if current_scope.parent == current_scope_id {
                break false;
            }

            current_scope_id = current_scope.parent;
        };

        if dup_found
            || self.functions.contains_key(name_str)
            || self.struct_ids.contains_key(name_str)
        {
            return Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::new(
                    Some(format!("name: {name_str} already used")),
                    name.start(),
                    name.len(),
                )],
                help = "Shadowing is not allowed in JPL",
                "Duplicate name found"
            ));
        }

        Ok(())
    }

    pub fn get_variable_type(&self, var: Span) -> miette::Result<&TypeVal> {
        let mut current_scope_id = self.cur_scope;
        let name = var.as_str(self.src);
        loop {
            let current_scope = &self.scopes[current_scope_id];
            if let Some(t) = current_scope.names.get(name) {
                return Ok(&t.var_type);
            }

            if current_scope.parent == current_scope_id {
                break;
            }

            current_scope_id = current_scope.parent;
        }

        if self.struct_ids.contains_key(name) {
            Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::new(
                    Some(format!("expected variable, found struct: {}", name)),
                    var.start(),
                    var.len(),
                )],
                "Invalid variable found"
            ))
        } else if self.functions.contains_key(name) {
            Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::new(
                    Some(format!("expected variable, found function: {}", name)),
                    var.start(),
                    var.len(),
                )],
                help = "Functions are not first class in JPL",
                "Invalid variable found"
            ))
        } else {
            Err(miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::new(
                    Some(format!("unkown variable: {}", name)),
                    var.start(),
                    var.len(),
                )],
                "Invalid variable found"
            ))
        }
    }

    pub fn functions(&self) -> &HashMap<&'a str, FunctionInfo<'a>> {
        &self.functions
    }

    pub fn type_size(&self, ty: &TypeVal) -> u64 {
        match ty {
            TypeVal::Int | TypeVal::Bool | TypeVal::Float => WORD_SIZE,
            TypeVal::Array(_, dims) => WORD_SIZE + WORD_SIZE * *dims as u64,
            TypeVal::Struct(id) => self.get_struct_id(*id).size,
            // IDK if this is correct.
            TypeVal::Void => 0,
        }
    }
}
