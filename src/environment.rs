use crate::{
    ast::{
        auxiliary::{Binding, LValue},
        types::Type,
    },
    typecheck::Typed,
    utils::Span,
};
use builtins::{builtin_fns, builtin_structs, builtin_vars};
use miette::{miette, LabeledSpan, Severity};
use std::collections::{hash_map::Entry, HashMap};

pub mod builtins;

pub struct Environment<'a> {
    src: &'a [u8],
    struct_ids: HashMap<&'a str, usize>,
    // Indexed by the struct id
    struct_info: Vec<StructInfo<'a>>,

    functions: HashMap<&'a str, FunctionInfo<'a>>,
    scopes: Vec<Scope<'a>>,
}

pub const GLOBAL_SCOPE_ID: usize = 0;

#[derive(Debug, Clone)]
pub struct StructInfo<'a> {
    fields: Box<[(&'a str, Typed)]>,
    id: usize,
    name: &'a str,
}

#[derive(Debug, Clone)]
pub struct FunctionInfo<'a> {
    args: Box<[Typed]>,
    ret: Typed,
    name: &'a str,
    scope: usize,
}

impl<'a> FunctionInfo<'a> {
    pub fn args(&self) -> &[Typed] {
        &self.args
    }

    pub fn ret(&self) -> &Typed {
        &self.ret
    }

    pub fn name(&self) -> &str {
        self.name
    }
}

/// Represents a all the variables in a scope in JPL.
#[derive(Debug, Clone)]
pub struct Scope<'a> {
    names: HashMap<&'a str, VarInfo<'a>>,

    /// The index of its parent scope in the vec of scopes. 0 is always the index of the global
    /// scope. The parent of the global scope will be itself
    parent: usize,
}

#[derive(Debug, Clone)]
pub struct VarInfo<'a> {
    var_type: Typed,
    // TODO: update once used
    bindings: Box<[&'a str]>,
}

impl<'a> StructInfo<'a> {
    pub fn fields(&self) -> &Box<[(&str, Typed)]> {
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
        }
    }

    pub fn add_struct(&mut self, name: Span, params: &[(Span, Type)]) -> miette::Result<()> {
        let name_str = name.as_str(self.src);
        match self.struct_ids.get(name_str) {
            Some(_) => {
                return Err(miette!(
                    severity = Severity::Error,
                    labels = vec![LabeledSpan::new(
                        Some(format!("Struct {} already declared", name_str)),
                        name.start(),
                        name.len(),
                    )],
                    "Redefinition of struct"
                ))
            }
            None => (),
        }

        let mut fields = Vec::with_capacity(params.len());
        let mut field_names = HashMap::with_capacity(params.len());

        for (field_name, field_type) in params {
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

            let converted_type = Typed::from_ast_type(field_type, self)?;
            entry.insert(*field_name);
            fields.push((field_name_str, converted_type))
        }

        let id = self.struct_info.len();
        self.struct_ids.insert(name_str, id);
        self.struct_info.push(StructInfo {
            fields: fields.into_boxed_slice(),
            id,
            name: name_str,
        });

        Ok(())
    }

    pub fn get_struct(&self, loc: Span) -> miette::Result<&StructInfo> {
        let id = self
            .struct_ids
            .get(loc.as_str(self.src))
            .map(|x| *x)
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

    pub fn src(&self) -> &[u8] {
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
        let scope = self.new_scope(GLOBAL_SCOPE_ID);

        self.check_name_free(name, scope)?;
        let name = name.as_str(self.src);

        let args = args
            .into_iter()
            .map(|arg| {
                let arg_type = Typed::from_ast_type(arg.variable_type(), self)?;
                self.add_lval(arg.l_value(), arg_type.clone(), scope)?;
                Ok(arg_type)
            })
            .collect::<miette::Result<Vec<_>>>()?
            .into_boxed_slice();
        let ret = Typed::from_ast_type(ret_type, self)?;

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

    pub fn new_scope(&mut self, parent_scope: usize) -> usize {
        self.scopes.push(Scope {
            names: HashMap::new(),
            parent: parent_scope,
        });

        self.scopes.len() - 1
    }

    pub fn add_lval(
        &mut self,
        l_val: &LValue,
        var_type: Typed,
        scope_id: usize,
    ) -> miette::Result<()> {
        self.check_name_free(l_val.variable(), scope_id)?;
        let name_str = l_val.variable().as_str(self.src);
        let bindings = l_val
            .array_bindings()
            .iter()
            .map(|t| t.iter())
            .flatten()
            .map(|s| s.as_str(self.src))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        self.scopes[scope_id].names.insert(
            name_str,
            VarInfo {
                var_type: var_type.clone(),
                bindings,
            },
        );

        for arr_len_binding in l_val.array_bindings().iter().map(|b| b.iter()).flatten() {
            self.add_lval(&LValue::from_span(*arr_len_binding), Typed::Int, scope_id)?;
        }

        Ok(())
    }

    fn check_name_free(&mut self, name: Span, scope_id: usize) -> miette::Result<()> {
        let mut current_scope_id = scope_id;
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

    pub fn get_variable_type(&self, var: Span, scope_id: usize) -> miette::Result<&Typed> {
        let mut current_scope_id = scope_id;
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
}
