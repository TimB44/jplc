TEST=test.jpl

all: run

compile: 
	cargo build


run: 
	cargo run -- $(TEST)

clean:
	rm -fr *.class
