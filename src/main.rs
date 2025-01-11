use clap::Parser;
use jplc::{cli::Args, compile};
use std::{fs::read_to_string, process::exit};
fn main() {
    let args = Args::parse();

    let source = read_to_string(&args.source).unwrap_or_else(|err| {
        eprintln!(
            "Error: can not open given source file {} - {}",
            args.source, err
        );
        exit(1);
    });

    compile(source, args.actions);
}
