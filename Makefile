TEST=test.jpl

all: run

compile: 
	cargo build -r

run: 
	./target/release/jplc -i $(TEST) 

clean:
	cargo clean	
