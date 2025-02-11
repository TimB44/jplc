use crate::{
    ast::types::{self, Type},
    utils::Span,
};
use miette::{miette, LabeledSpan, Severity};
use std::collections::{hash_map::Entry, HashMap};

pub struct Environment<'a> {
    src: &'a [u8],
    struct_ids: HashMap<&'a str, usize>,
    // Indexed by the struct id
    struct_info: Vec<StructInfo<'a>>,
}

pub struct StructInfo<'a> {
    fields: Box<[(&'a str, Typed)]>,
    id: usize,
}

impl<'a> StructInfo<'a> {
    pub fn fields(&self) -> &Box<[(&str, Typed)]> {
        &self.fields
    }

    pub fn id(&self) -> usize {
        self.id
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
        }];

        Self {
            struct_ids,
            struct_info: struct_fields,
            src,
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

    //pub fn get_struct_info(&self, id: usize) -> &StructInfo {
    //    &self.struct_info[id]
    //}

    pub fn src(&self) -> &[u8] {
        self.src
    }
}

pub trait GetType {
    fn get_type(&self, env: &Environment) -> miette::Result<Typed>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Typed {
    Int,
    Bool,
    Float,

    // Type of the array and the rank
    Array(Box<Typed>, u8),

    // Id of struct
    Struct(usize),
    Void,
}

pub trait TypeState {}

#[derive(Debug, Clone)]
pub struct UnTyped {}
impl TypeState for UnTyped {}
impl TypeState for Typed {}

impl Typed {
    pub fn as_str(&self, env: &Environment) -> String {
        match self {
            Typed::Int => "int".to_string(),
            Typed::Bool => "bool".to_string(),
            Typed::Float => "float".to_string(),
            Typed::Array(element_type, rank) => {
                format!(
                    "{}[{}]",
                    element_type.as_str(env),
                    ",".repeat((*rank as usize) - 1)
                )
            }
            // TODO: not ideal
            Typed::Struct(id) => env
                .struct_ids
                .iter()
                .filter(|(_, v)| *v == id)
                .next()
                .unwrap()
                .0
                .to_string(),
            Typed::Void => "void".to_string(),
        }
    }
}

impl Typed {
    pub fn from_ast_type(t: &Type, env: &Environment) -> miette::Result<Self> {
        Ok(match t.kind() {
            types::TypeKind::Int => Typed::Int,
            types::TypeKind::Bool => Typed::Bool,
            types::TypeKind::Float => Typed::Float,
            types::TypeKind::Array(inner, rank) => {
                Typed::Array(Box::new(Self::from_ast_type(inner, env)?), *rank)
            }
            types::TypeKind::Struct => {
                let id = env.get_struct(t.location())?.id;

                Typed::Struct(id)
            }
            types::TypeKind::Void => Typed::Void,
        })
    }
}
