TEST=test.jpl

all: run

compile: 
	cargo build

run: 
	cargo run -- -p $(TEST) 

clean:
	cargo clean	
