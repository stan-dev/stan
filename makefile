# g++ (GCC), clang (Clang)
CC = clang++ # g++
OPT = -O3 -Wall -g  #-rdynamic
INCLUDES = -I src -I lib
CFLAGS = $(OPT) $(INCLUDES)
CFLAGS_T = $(CFLAGS) -I lib/gtest/include  -I lib/gtest # -lpthread

# find all unit tests
UNIT_TESTS := $(wildcard src/test/*/*.cpp)
UNIT_TEST_OBJ := $(UNIT_TESTS:src/test/%_test.cpp=test/%)

# DEFAULT
# =========================================================

.PHONY: all test-all
all: test-all

# TEST
# =========================================================
.PHONY: tmp
tmp:
	$(foreach var,$(UNIT_TEST_OBJS), $(var);)


test:
	mkdir -p ar test test/agrad test/io test/maths test/mcmc\
	    test/memory test/prob test/maths

ar/libgtest.a:  | test
	$(CC) $(CFLAGS_T) -c lib/gtest/src/gtest-all.cc -o ar/gtest-all.o
	ar -rv ar/libgtest.a ar/gtest-all.o

# : src/stan/%.hpp
test/% : src/test/%_test.cpp ar/libgtest.a
	$(CC) $(CFLAGS_T) src/$@_test.cpp lib/gtest/src/gtest_main.cc ar/libgtest.a -o $@

# run all tests
test-all: $(UNIT_TESTS_OBJ)
	$(foreach var,$(UNIT_TESTS_OBJ), $(var);)




# DEMO
# =========================================================

demo:
	mkdir -p demo

demo/% : src/demo/%.cpp | demo
	$(CC) $(CFLAGS) src/$@.cpp -c -o $@.o
	$(CC) $(CFLAGS) $@.o -o $@

demo-all: demo/bivar_norm demo/model1 demo/eight_schools


# DOC
# =========================================================

.PHONY: dox
dox:
	mkdir -p doc/api

doxygen: | dox
	doxygen doc/doxygen.cfg


# CLEAN
# =========================================================

.PHONY: clean clean-dox clean-all
clean:
	rm -rf demo test ar *.dSYM

clean-dox:
	rm -rf doc/api

clean-all: clean clean-dox
