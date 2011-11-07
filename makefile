# g++ (GCC), clang (Clang)
CC = g++ # clang++ # g++
OPT = -O3 -Wall -g  #-rdynamic
INCLUDES = -I src -I lib
CFLAGS = $(OPT) $(INCLUDES)
CFLAGS_T = $(CFLAGS) -I lib/gtest/include  -I lib/gtest # -lpthread

# DEFAULT
# =========================================================

all: test-all


# DEMO
# =========================================================

demo:
	mkdir -p demo

demo/% : src/demo/%.cpp | demo
	$(CC) $(CFLAGS) src/$@.cpp -c -o $@.o
	$(CC) $(CFLAGS) $@.o -o $@

demo-all: demo/bivar_norm demo/model1 demo/eight_schools


# TEST
# =========================================================

test:
	mkdir -p ar test test/agrad test/io test/maths test/mcmc test/memory test/prob test/maths

ar/libgtest.a:  | test
	$(CC) $(CFLAGS_T) -c lib/gtest/src/gtest-all.cc -o ar/gtest-all.o
	ar -rv ar/libgtest.a ar/gtest-all.o

# : src/stan/%.hpp
test/% : src/test/%_test.cpp ar/libgtest.a
	$(CC) $(CFLAGS_T) src/$@_test.cpp lib/gtest/src/gtest_main.cc ar/libgtest.a -o $@

test-all: test/agrad/agrad test/agrad/agrad_special_functions test/agrad/agrad_eigen test/agrad/matrix test/io/reader test/io/dump test/maths/special_functions test/maths/matrix test/memory/stack_alloc test/prob/distributions test/prob/online_avg test/prob/rhat test/prob/transform
	test/agrad/agrad 
	test/agrad/agrad_special_functions
	test/agrad/agrad_eigen
	test/agrad/matrix
	test/io/reader
	test/io/dump
	test/maths/matrix
	test/maths/special_functions 
	test/memory/stack_alloc 
	test/prob/distributions 
	test/prob/online_avg
	test/prob/rhat
	test/prob/transform



# DOC
# =========================================================

dox:
	mkdir -p doc/api

doxygen: | dox
	doxygen doc/doxygen.cfg


# CLEAN
# =========================================================

clean:
	rm -r -f demo test ar *.dSYM

clean-all:
	rm -r -f demo test ar *.dSYM doc/api
