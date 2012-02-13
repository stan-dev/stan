##
# Makefile for Stan.
# This makefile relies heavily on the make defaults for
# make 3.81.
##

# The default target of this Makefile is...
help:

##
# Users should only need to set these three variables for use.
# - CC: The compiler to use. Expecting g++ or clang++.
# - O: Optimization level. Valid values are {0, 1, 2, 3}.
# - OS: {mac, win, linux}. 
##
CC = g++
O = 0
# OS is set automatically by this script
-include src/makefile/detect_os

##
# Get information about the compiler used.
# - CC_TYPE: {g++, clang++, other}
# - CC_MAJOR: major version of CC
# - CC_MINOR: minor version of CC
##
-include src/makefile/detect_cc
# FIXME: verify compiler

##
# Set default compiler options.
## 
CFLAGS = -I src -I lib
CFLAGS += -O$O
CFLAGS += -Wall
CFLAGS_GTEST = -I lib/gtest/include -I lib/gtest
LIBGTEST = test/gtest.o
GTEST_MAIN = lib/gtest/src/gtest_main.cc
EXE = 

##
# These includes should update the following variables
# based on the OS:
#   - CFLAGS
#   - CFLAGS_GTEST
#   - EXE
##
-include src/makefile/$(OS)

#%.d : src/%.cpp
#	@echo $(dir $@)
#mkdir -p $(dir $@)

.PHONY: help
help:
	@echo '------------------------------------------------------------'
	@echo 'Stan: makefile'
	@echo '  Current configuration:'
	@echo '  - OS (Operating System):' $(OS)
	@echo '  - CC (Compiler):        ' $(CC)
	@echo '  - O (Optimize Level):   ' $(O)
	@echo 'Available targets: '
	@echo '  Tests:'
	@echo '  - test-unit:   Runs unit tests.'
	@echo '  - test-models: Runs diagnostic models.'
	@echo '  - test-bugs:   Runs the bugs examples'
	@echo '  - test-all:    Runs all tests.'
	@echo '  Clean:'
	@echo '  - clean:       Basic clean. Leaves doc and compiled'
	@echo '                 libraries intact.'
	@echo '  - clean-all:   Cleans up all of Stan.'
	@echo '------------------------------------------------------------'

##
# All testing related make commands.
##
-include src/makefile/tests

##
# All model building related make commands.
##
-include src/makefile/models

##
# All demo related make commands.
##
-include src/makefile/demo

##
# Clean up.
##
.PHONY: clean clean-models clean-dox clean-demo clean-all
clean:
	$(RM) -r *.dSYM

clean-models:
	$(RM) -r models

clean-dox:
	$(RM) -r doc/api

clean-demo:
	$(RM) -r demo

clean-all: clean clean-models clean-dox clean-demo
	$(RM) -r test 
