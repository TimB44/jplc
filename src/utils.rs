use std::process::exit;

pub fn exit_with_error(err: miette::Error) -> ! {
    println!("{:?}", err);
    println!("Compilation failed");
    exit(1);
}
