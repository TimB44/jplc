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
