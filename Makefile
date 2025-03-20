TEST=test.jpl

all: run

compile: 
	cargo build -r

build: compile

run: 
	./target/release/jplc -s $(TEST) 

clean:
	cargo clean	

