TEST=test.jpl

all: run

compile: 
	cargo build -r

run: 
	./target/release/jplc -t $(TEST) 

clean:
	cargo clean	
