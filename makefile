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
# - AR: archiver (must specify for cross-compiling)
# - OS: {mac, win, linux}. 
##
CC = g++
O = 3
AR = ar
# OS is set automatically by this script
-include make/detect_os

##
# Get information about the compiler used.
# - CC_TYPE: {g++, clang++, mingw32-g++, other}
# - CC_MAJOR: major version of CC
# - CC_MINOR: minor version of CC
##
-include make/detect_cc
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
LDLIBS = -Lbin -lstan

##
# Tell make the default way to compile a .o file.
##
%.o : %.cpp
	$(COMPILE.c) $(OUTPUT_OPTION) $<

##
# Tell make the default way to compile a .o file.
##
bin/%.o : src/%.cpp
	@mkdir -p $(dir $@)
	$(COMPILE.c) $(OUTPUT_OPTION) $<


##
# These includes should update the following variables
# based on the OS:
#   - CFLAGS
#   - CFLAGS_GTEST
#   - EXE
##
-include make/$(OS)

.PHONY: help
help:
	@echo '------------------------------------------------------------'
	@echo 'Stan: makefile'
	@echo '  Current configuration:'
	@echo '  - OS (Operating System):   ' $(OS)
	@echo '  - CC (Compiler):           ' $(CC)
	@echo '  - O (Optimize Level):      ' $(O)
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


-include make/libstan  # libstan.a
-include make/tests    # tests: test-all, test-unit, test-models
-include make/models   # models
-include make/command  # bin/stanc
-include make/doxygen  # doxygen
-include make/dist     # dist: for distribution
-include make/manual   # manual: manual, doc/stan-reference.pdf
-include make/demo     # for building demos


#%.d : src/%.cpp
#	@echo $(dir $@)
#mkdir -p $(dir $@)

##
# Rule for generating dependencies.
# Applies to all *.cpp files in src.
# Test cpp files are handled slightly differently.
##
%.d : src/%.cpp
	@if test -d $(dir $@); \
	then \
	(set -e; \
	rm -f $@; \
	$(CC) $(CFLAGS) $(TARGET_ARCH) -MM $< > $@.$$$$; \
	sed -e 's,\($(notdir $*)\)\.o[ :]*,$(dir $@)\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$);\
	fi

##
# Clean up.
##
.PHONY: clean clean-demo clean-dox clean-manual clean-models clean-all
clean:
	$(RM) -r *.dSYM
	$(RM) $(OFILES) bin/libstan.a

clean-demo:
	$(RM) -r demo

clean-dox:
	$(RM) -r doc/api

clean-manual:
	cd src/docs/stan-reference; $(RM) *.aux *.bbl *.blg *.log *.toc *.pdf
	$(RM) doc/stan-reference.pdf

clean-models:
	$(RM) -r models $(MODEL_HEADER).gch $(MODEL_HEADER).pch

clean-all: clean clean-models clean-dox clean-demo
	$(RM) -r test bin

