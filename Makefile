TEST=test.jpl
.PHONY: all compile run clean test

all: run


compile: 
	cargo build 

run: 
	./target/debug/jplc $(FLAGS) $(TEST) 

clean:
	cargo clean	

test: compile
	cd grader && make | head -n 50 


