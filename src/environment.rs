use std::collections::{hash_map::Entry, HashMap};
mod builtins;

use miette::{miette, LabeledSpan, Severity};

use crate::{ast::types::Type, typecheck::Typed, utils::Span};

pub struct Environment<'a> {
    src: &'a [u8],
    struct_ids: HashMap<&'a str, usize>,
    // Indexed by the struct id
    struct_info: Vec<StructInfo<'a>>,

    functions: HashMap<&'a str, FunctionInfo<'a>>,
    scopes: Vec<Scope<'a>>,
}

pub struct StructInfo<'a> {
    fields: Box<[(&'a str, Typed)]>,
    id: usize,
    name: &'a str,
}
pub struct FunctionInfo<'a> {
    args: Box<[Typed]>,
    ret: Typed,
    name: &'a str,
}

pub struct Scope<'a> {
    names: HashMap<&'a str, Option<Typed>>,

    /// The index of its parent scope in the vec of scopes. 0 is always the index of the global
    /// scope. The parent of the global scope will be itself
    parent: usize,
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
        let struct_ids = HashMap::from([("rgba", 0)]);
        let struct_fields = vec![StructInfo {
            fields: vec![
                ("r", Typed::Float),
                ("g", Typed::Float),
                ("b", Typed::Float),
                ("a", Typed::Float),
            ]
            .into_boxed_slice(),
            id: 0,
            name: "rgba",
        }];

        // Of one float argument, returning a float: sqrt, exp, sin, cos, tan, asin, acos, atan, and log
        // Of two float arguments, returning a float: pow and atan2
        // /////////////////////The to_float function, which converts an int to a float
        // The to_int function, which converts a float to an int, with positive and negative infinity converting into the maximum and minimum integers, and NaN converting to 0.
        Self {
            struct_ids,
            struct_info: struct_fields,
            src,
            functions: todo!(),
            scopes: todo!(),
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
}
