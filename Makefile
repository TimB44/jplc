TEST=test.jpl

all: run

compile: 
	cargo build 

build: compile

run: 
	./target/debug/jplc -s $(TEST) 

clean:
	cargo clean	

test: compile
	cd grader && make | head -n 50 


