//! Command line interface to the JPL compiler

use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "jplc")]
#[command(about, version)]
pub struct Args {
    // Source filename to compile
    #[arg(required = true)]
    pub source: String,

    #[clap(flatten)]
    pub actions: Mode,

    #[clap(short = 'O', value_parser, default_value_t = OptLevel::None)]
    pub opt: OptLevel,
}

#[derive(Debug, Clone, clap::Args)]
#[group(required = false, multiple = false)]
pub struct Mode {
    /// Output Lexed Tokens only
    #[clap(short, long)]
    pub lex: bool,

    /// Output the parsed AST  
    #[clap(short, long)]
    pub parse: bool,

    /// Perform type checking  
    #[clap(short, long)]
    pub typecheck: bool,

    /// Generate strait-line C code
    #[clap(short = 'i', long = "c-ir")]
    pub c_ir: bool,

    /// Generate assembly code
    #[clap(short = 's', long = "asm")]
    pub assembly: bool,
}

use std::{fmt::Display, str::FromStr};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    None,
    O1,
    O3,
}

impl FromStr for OptLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "0" => Ok(OptLevel::None),
            "1" => Ok(OptLevel::O1),
            "2" => Ok(OptLevel::O1),
            "3" => Ok(OptLevel::O3),
            _ => Err(format!("invalid optimization level '{}'", s)),
        }
    }
}

impl Display for OptLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptLevel::None => write!(f, "0"),
            OptLevel::O1 => write!(f, "1"),
            OptLevel::O3 => write!(f, "3"),
        }
    }
}
