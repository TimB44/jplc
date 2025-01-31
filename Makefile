TEST=test.jpl

all: run

compile: 
	cargo build -r

run: 
	./target/release/jplc -p $(TEST) 

clean:
	cargo clean	
