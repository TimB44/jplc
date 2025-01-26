TEST=test.jpl

all: run

compile: 
	cargo build 

run: 
	cargo run -q -- -p $(TEST) 

clean:
	cargo clean	
