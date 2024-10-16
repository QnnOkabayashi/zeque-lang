use crate::{ast, util::Range};
use index_vec::IndexVec;
use miette::{Diagnostic, LabeledSpan};
use smol_str::SmolStr;
use std::iter;
use thiserror::Error;

index_vec::define_index_type! {
    pub struct ErrorIdx = u32;
}

pub type ErrorVec = IndexVec<ErrorIdx, NameResolutionError>;

#[derive(Clone, Debug, Error)]
#[error("name '{name}' not found")]
pub struct NameResolutionError {
    name: SmolStr,
    range: Range,
}

impl NameResolutionError {
    pub fn new(name: ast::Name) -> Self {
        NameResolutionError {
            name: name.text,
            range: name.range,
        }
    }
}

impl Diagnostic for NameResolutionError {
    fn help(&self) -> Option<Box<dyn std::fmt::Display + '_>> {
        Some(Box::new("use a name that's in scope"))
    }

    fn labels(&self) -> Option<Box<dyn Iterator<Item = LabeledSpan> + '_>> {
        Some(Box::new(iter::once(LabeledSpan::at(
            self.range,
            format!("name '{}' is not valid in this scope", self.name),
        ))))
    }
}
