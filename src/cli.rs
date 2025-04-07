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

// #[derive(Debug, Clone, clap::Args)]
// #[group(required = false, multiple = false)]
// pub struct OptLevelArgs {
//     /// Basic assembly optimizations
//     #[clap(long = "O1")]
//     pub o1: bool,
// }

use std::{fmt::Display, str::FromStr};

impl FromStr for OptLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "0" => Ok(OptLevel::None),
            "1" => Ok(OptLevel::O1),
            // "2" | "O2" | "o2" => Ok(OptLevel::O2),
            // "3" | "O3" | "o3" => Ok(OptLevel::O3),
            _ => Err(format!("invalid optimization level '{}'", s)),
        }
    }
}

#[derive(Debug, Clone)]
pub enum OptLevel {
    None,
    O1,
}

impl Display for OptLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptLevel::None => write!(f, "0"),
            OptLevel::O1 => write!(f, "1"),
        }
    }
}
