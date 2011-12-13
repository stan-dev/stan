# g++ (GCC), clang (Clang)
CC = clang++
EIGEN_OPT = -DNDEBUG
OPTIMIZE_OPT = 0
OPT = -O$(OPTIMIZE_OPT) -Wall -g  $(EIGEN_OPT) #-rdynamic 

INCLUDES = -I src -I lib
INCLUDES_T = -I lib/gtest/include  -I lib/gtest
CFLAGS = $(OPT) $(INCLUDES)
ifneq (,$(findstring g++,$(CC)))
	CFLAGS += -std=gnu++0x
endif
CFLAGS_T = $(CFLAGS) $(INCLUDES_T) -DGTEST_HAS_PTHREAD=0


# find all unit tests
UNIT_TESTS := $(wildcard src/test/*/*.cpp)
UNIT_TESTS_DIR := $(sort $(dir $(UNIT_TESTS)))
UNIT_TESTS_OBJ := $(UNIT_TESTS:src/test/%_test.cpp=test/%)

# DEFAULT
# =========================================================

.PHONY: all test-all
all: test-all

# TEST
# =========================================================

test:
	mkdir -p ar test 
	$(foreach var,$(UNIT_TESTS_DIR:src/%/=%), mkdir -p $(var);)

ar/libgtest.a:  | test
	$(CC) $(CFLAGS_T) -c lib/gtest/src/gtest-all.cc -o ar/gtest-all.o
	ar -rv ar/libgtest.a ar/gtest-all.o


# The last argument, $$(wildcard src/stan/$$(dir $$*)*.hpp), puts *.hpp files from the
#   same directory as a prerequisite. For example, for test/prob/distributions, it will expand to
#   all the hpp files in the src/stan/prob/ directory.
.SECONDEXPANSION:
test/% : src/test/%_test.cpp ar/libgtest.a $$(wildcard src/stan/$$(dir $$*)*.hpp)
	$(CC) $(CFLAGS_T) src/$@_test.cpp lib/gtest/src/gtest_main.cc ar/libgtest.a -o $@
	-$@ --gtest_output="xml:$@.xml"

# run all tests
test-all: $(UNIT_TESTS_OBJ)
	-$(foreach var,$(UNIT_TESTS_OBJ), $(var) --gtest_output="xml:$(var).xml";)


# MODELS (to be passed through demo/gm)
# =========================================================

models:
	mkdir -p models

models/% :  | demo/gm models
	@echo '--- translating src/models/%.stan using demo/gm ---'
	cat src/models/$@.stan | demo/gm > models/$(notdir $@).cpp
	@echo '--- building models/$(notdir $@).cpp ---'
	$(CC) $(CFLAGS) models/$@.cpp -c -o models/$@.o
	$(CC) $(CFLAGS) models/$@.o -o models/$@
	@echo '--- copying model data to models/ ---'
	cp src/models/%@*data models/

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

.PHONY: dox doxygen
dox:
	mkdir -p doc/api

doxygen: | dox
	doxygen doc/doxygen.cfg


# CLEAN
# =========================================================

.PHONY: clean clean-dox clean-all
clean:
	rm -rf demo test *.dSYM

clean-models:
	rm -rf models

clean-dox:
	rm -rf doc/api

clean-all: clean clean-dox clean-models
	rm -rf ar
