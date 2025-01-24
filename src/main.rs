use clap::Parser;
use jplc::{cli::Args, compile, utils::exit_with_error};
use miette::{miette, LabeledSpan, NamedSource, Severity};
use std::fs::read;

fn main() {
    let args = Args::parse();

    let bytes = read(&args.source).unwrap_or_else(|err| {
        dbg!(&err);
        exit_with_error(miette!(
            severity = Severity::Error,
            help = format!(
                "Ensure the file '{}' exists and you have the necessary permissions to read it",
                args.source
            ),
            "Cannot open source file\nError: {}",
            err
        ));
    });

    let source = String::from_utf8(bytes).unwrap_or_else(|err| {
        let err_loc = err.utf8_error().valid_up_to();
        let bytes = err.into_bytes();
        exit_with_error(
            miette!(
                severity = Severity::Error,
                labels = vec![LabeledSpan::at_offset(err_loc, "Invalid UTF-8 found here"),],
                help = "Code must be encoded as ASCII",
                "Source file contains invalid character"
            )
            .with_source_code(NamedSource::new(&args.source, bytes)),
        );
    });

    compile(args.source, source, args.actions);
}
