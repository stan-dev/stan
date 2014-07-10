# Makefile for Stan.
# This makefile relies heavily on the make defaults for
# make 3.81.
##


# The default target of this Makefile is...
help:

## Disable implicit rules.
SUFIXES:

##
# Users should only need to set these three variables for use.
# - CC: The compiler to use. Expecting g++ or clang++.
# - O: Optimization level. Valid values are {0, 1, 2, 3}.
# - AR: archiver (must specify for cross-compiling)
# - OS: {mac, win, linux}
# - C++11: Compile with C++11 extensions, Valid values: {true, false}. 
##
CC = g++
O = 3
O_STANC = 0
AR = ar
C++11 = false

##
# Library locations
##
STAN_HOME := $(dir $(firstword $(MAKEFILE_LIST)))
EIGEN ?= lib/eigen_3.2.0
BOOST ?= lib/boost_1.54.0
GTEST ?= lib/gtest_1.7.0

##
# Set default compiler options.
## 
CFLAGS = -I src -isystem $(EIGEN) -isystem $(BOOST) -Wall -DBOOST_RESULT_OF_USE_TR1 -DBOOST_NO_DECLTYPE -DBOOST_DISABLE_ASSERTS -pipe -DEIGEN_NO_DEBUG
CFLAGS_GTEST = -DGTEST_USE_OWN_TR1_TUPLE
LDLIBS = -Lbin -lstan
LDLIBS_STANC = -Lbin -lstanc
EXE = 
WINE =
PATH_SEPARATOR = /


##
# Get information about the compiler used.
# - CC_TYPE: {g++, clang++, mingw32-g++, other}
# - CC_MAJOR: major version of CC
# - CC_MINOR: minor version of CC
##
-include make/detect_cc
# FIXME: verify compiler

# OS is set automatically by this script
##
# These includes should update the following variables
# based on the OS:
#   - CFLAGS
#   - CFLAGS_GTEST
#   - EXE
##
-include make/detect_os

##
# Get information about the version of make.
##
-include make/detect_make

##
# Tell make the default way to compile a .o file.
##
%.o : %.cpp
	$(COMPILE.c) -O$O $(OUTPUT_OPTION) $<

##
# Tell make the default way to compile a .o file.
##
bin/%.o : src/%.cpp
	@mkdir -p $(dir $@)
	$(COMPILE.c) -O$O $(OUTPUT_OPTION) $<

##
# Rule for generating dependencies.
# Applies to all *.cpp files in src.
# Test cpp files are handled slightly differently.
##
bin/%.d : src/%.cpp
	@if test -d $(dir $@); \
	then \
	(set -e; \
	rm -f $@; \
	$(CC) $(CFLAGS) -O$O $(TARGET_ARCH) -MM $< > $@.$$$$; \
	sed -e 's,\($(notdir $*)\)\.o[ :]*,$(dir $@)\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$);\
	fi

%.d : %.cpp
	@if test -d $(dir $@); \
	then \
	(set -e; \
	rm -f $@; \
	$(CC) $(CFLAGS) -O$O $(TARGET_ARCH) -MM $< > $@.$$$$; \
	sed -e 's,\($(notdir $*)\)\.o[ :]*,$(dir $@)\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$);\
	fi


.PHONY: help
help:
	@echo '--------------------------------------------------------------------------------'
	@echo 'Stan makefile:'
	@echo '  Current configuration:'
	@echo '  - OS (Operating System):   ' $(OS)
	@echo '  - CC (Compiler):           ' $(CC)
	@echo '  - Compiler version:        ' $(CC_MAJOR).$(CC_MINOR)
	@echo '  - O (Optimization Level):  ' $(O)
	@echo '  - O_STANC (Opt for stanc): ' $(O_STANC)
ifdef TEMPLATE_DEPTH
	@echo '  - TEMPLATE_DEPTH:          ' $(TEMPLATE_DEPTH)
endif
	@echo '  - STAN_HOME                ' $(STAN_HOME)
	@echo '  Library configuration:'
	@echo '  - EIGEN                    ' $(EIGEN)
	@echo '  - BOOST                    ' $(BOOST)
	@echo '  - GTEST                    ' $(GTEST)
	@echo ''
	@echo 'Common targets:'
	@echo '  Model related:'
	@echo '  - bin/libstan.a  : Build the Stan static library (used in linking models).'
	@echo '  - bin/libstanc.a : Build the Stan compiler static library'
	@echo '  Documentation:'
	@echo '  - manual         : Builds the reference manual. Copies built manual to'
	@echo '                     doc/stan-reference-$(VERSION_STRING).pdf'
	@echo '                     (requires LaTeX installation)'
	@echo '  - doxygen        : Builds the API documentation. The documentation is located'
	@echo '                     doc/api/'
	@echo '                     (requires doxygen installation)'
	@echo '  TESTS (requires make 3.81 or higher):'
	@echo ''
	@echo '    All Tests'
	@echo '      - test-all'
	@echo ''
	@echo '    Header Tests'
	@echo '      - test-headers'
	@echo ''
	@echo '    Unit Tests'
	@echo '      The unit tests are broken into three targets:'
	@echo '        - src/test/unit'
	@echo '        - src/test/unit-agrad-rev'
	@echo '        - src/test/unit-agrad-fwd'
	@echo ''
	@echo '      Subdirectory of Unit Tests'
	@echo '        For example, to run the unit tests under meta'
	@echo '          - src/test/unit/meta'
	@echo ''
	@echo '      Single Unit Test'
	@echo '        For example, to run the diag_post_multiply test, make the target'
	@echo '          - test/unit-agrad-fwd/matrix/diag_post_multiply$(EXE)'
	@echo ''
	@echo '  Clean:'
	@echo '  - clean          : Basic clean. Leaves doc and compiled libraries intact.'
	@echo '  - clean-all      : Cleans up all of Stan.'
	@echo '  Higher level targets:'
	@echo '  - docs           : Builds all docs.'
	@echo ''
	@echo '  Warning: Deprecated test targets'
	@echo '  - test-unit      : Runs unit tests.'
	@echo '    This has been split into 3 separate targets:'
	@echo '      src/test/unit'
	@echo '      src/test/unit-agrad-rev'
	@echo '      src/test/unit-agrad-fwd'
	@echo '  - test-distributions : Runs unit tests for the distributions'
	@echo '    Use this target instead: src/test/unit-distribution'
	@echo '  - test-models    : Runs diagnostic models.'
	@echo '    Run this target from CmdStan: src/test/models'
	@echo '  - test-bugs      : Runs the bugs examples (subset of test-models).'
	@echo '    Run this target from CmdStan: src/test/models/bugs_examples'
	@echo ''
	@echo '--------------------------------------------------------------------------------'

-include make/libstan  # libstan.a
-include make/tests    # tests: test-all, test-unit, test-models
-include make/models   # models
-include make/doxygen  # doxygen
-include make/manual   # manual: manual, doc/stan-reference.pdf
-include make/demo     # for building demos
-include make/local    # for local stuff

ifneq (,$(filter-out runtest/%,$(filter-out clean%,$(MAKECMDGOALS))))
  -include $(addsuffix .d,$(subst $(EXE),,$(MAKECMDGOALS)))
endif

ifneq (,$(filter runtest/%,$(MAKECMDGOALS)))
  -include $(addsuffix .d,$(subst runtest/,,$(MAKECMDGOALS)))
endif

ifneq (,$(filter runtest_no_fail/%,$(MAKECMDGOALS)))
  -include $(addsuffix .d,$(subst runtest_no_fail/,,$(MAKECMDGOALS)))
endif

.PHONY: docs
docs: manual doxygen

##
# Clean up.
##
MODEL_SPECS := $(shell find src/test -type f -name '*.stan')
.PHONY: clean clean-demo clean-dox clean-manual clean-models clean-all
clean:
	$(RM) $(shell find src -type f -name '*.dSYM') $(shell find src -type f -name '*.d.*')
	$(RM) $(wildcard $(MODEL_SPECS:%.stan=%.cpp) $(MODEL_SPECS:%.stan=%$(EXE)) $(MODEL_SPECS:%.stan=%.o) $(MODEL_SPECS:%.stan=%.d))

clean-dox:
	$(RM) -r doc/api

clean-manual:
	cd src/docs/stan-reference; $(RM) *.brf *.aux *.bbl *.blg *.log *.toc *.pdf *.out *.idx *.ilg *.ind *.cb *.cb2 *.upa

clean-models:
	$(RM) -r models $(MODEL_HEADER).d

clean-all: clean clean-manual clean-models
	$(RM) -r test/* bin
	$(RM) $(shell find src -type f -name '*.d') $(shell find src -type f -name '*.o') $(shell find src/test/unit-distribution -name '*_generated_test.cpp' -type f | sed 's#\(.*\)/.*#\1/*_generated_test.cpp#' | sort -u)

