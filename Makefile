TEST=test.jpl

all: run

compile: 
	cargo build

run: 
	cargo run -- -l $(TEST) 

clean:
	rm -fr *.class
