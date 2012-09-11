# Makefile for Stan.
# This makefile relies heavily on the make defaults for
# make 3.81.
##

## Disable implicit rules.
SUFFIXES:

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

##
# Library locations
##
EIGEN ?= lib/eigen_3.1.1
BOOST ?= lib/boost_1.51.0
GTEST ?= lib/gtest_1.6.0

##
# Set default compiler options.
## 
CFLAGS = -I src -I $(EIGEN) -I $(BOOST) -O$O -Wall
LDLIBS = -Lbin -lstan
LDLIBS_STANC = -Lbin -lstanc
EXE = 
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
#   - PCH
##
-include make/os_detect

%$(EXE) : %.o %.cpp bin/libstan.a
	$(LINK.c) $(OUTPUT_OPTION) $< $(LDLIBS)

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
# Rule for generating dependencies.
# Applies to all *.cpp files in src.
# Test cpp files are handled slightly differently.
##
bin/%.d : src/%.cpp
	@if test -d $(dir $@); \
	then \
	(set -e; \
	rm -f $@; \
	$(CC) $(CFLAGS) $(TARGET_ARCH) -MM $< > $@.$$$$; \
	sed -e 's,\($(notdir $*)\)\.o[ :]*,$(dir $@)\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$);\
	fi

%.d : %.cpp
	@if test -d $(dir $@); \
	then \
	(set -e; \
	rm -f $@; \
	$(CC) $(CFLAGS) $(TARGET_ARCH) -MM $< > $@.$$$$; \
	sed -e 's,\($(notdir $*)\)\.o[ :]*,$(dir $@)\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$);\
	fi


.PHONY: help
help:
	@echo '--------------------------------------------------------------------------------'
	@echo 'Stan: makefile'
	@echo '  Current configuration:'
	@echo '  - OS (Operating System):   ' $(OS)
	@echo '  - CC (Compiler):           ' $(CC)
	@echo '  - O (Optimize Level):      ' $(O)
	@echo ''
	@echo 'Build a Stan model:'
	@echo '  Given a Stan model at foo/bar.stan, the make target is:'
	@echo '  - foo/bar$(EXE)'
	@echo ''
	@echo '  This target will:'
	@echo '  1. Build the Stan compiler: bin/stanc$(EXE).'
	@echo '  2. Use the Stan compiler to generate C++ code, foo/bar.cpp.'
	@echo '  3. Compile the C++ code using $(CC) to generate foo/bar$(EXE)'
	@echo ''
	@echo '  Example - Sample from a normal:'
	@echo '    1. Copy src/models/basic_distributions/normal.stan to foo/normal.stan:'
	@echo '       mkdir foo'
	@echo '       cp src/models/basic_distributions/normal.stan foo'
	@echo '    2. Build the model foo/normal$(EXE):'
	@echo '       make foo/normal$(EXE)'
	@echo '    3. Run the model:'
	@echo '       foo'$(PATH_SEPARATOR)'normal$(EXE) --samples=foo/normal.csv'
	@echo '    4. Look at the samples:'
	@echo '       more foo'$(PATH_SEPARATOR)'normal.csv'
	@echo ''
	@echo 'Common targets:'
	@echo '  Model related:'
	@echo '  - bin/stanc$(EXE): Build the Stan compiler.'
	@echo '  - bin/libstan.a  : Build the Stan static library (used in linking models).'
	@echo '  - bin/libstanc.a : Build the Stan compiler static library (used in linking'
	@echo '                     bin/stanc$(EXE))'
	@echo '  - models/*$(EXE) : If a Stan model exists at src/models/*.stan, this target'
	@echo '                     will copy the Stan model to models/*.stan, then build the'
	@echo '                     Stan model.'
	@echo '  - *$(EXE)        : If a Stan model exists at *.stan, this target will build'
	@echo '                     the Stan model as an executable.'
	@echo '  Tests:'
	@echo '  - test-unit      : Runs unit tests.'
	@echo '  - test-models    : Runs diagnostic models.'
	@echo '  - test-bugs      : Runs the bugs examples (subset of test-models).'
	@echo '  - test-all       : Runs all tests.'
	@echo '  Documentation:'
	@echo '  - manual         : Builds the reference manual. Copies built manual to'
	@echo '                     doc/stan-reference.pdf'
	@echo '  - doxygen        : Builds the API documentation. The documentation is located'
	@echo '                     doc/api/'
	@echo '  Distribution:'
	@echo '  - dist           : Creates a tarball for distribution. The resulting tarball is'
	@echo '                     created at the top level as stan-src-<version>.tgz.'
	@echo '  Clean:'
	@echo '  - clean          : Basic clean. Leaves doc and compiled libraries intact.'
	@echo '  - clean-all      : Cleans up all of Stan.'
	@echo '--------------------------------------------------------------------------------'

-include make/libstan  # libstan.a
-include make/tests    # tests: test-all, test-unit, test-models
-include make/models   # models
-include make/command  # bin/stanc
-include make/doxygen  # doxygen
-include make/dist     # dist: for distribution
-include make/manual   # manual: manual, doc/stan-reference.pdf
-include make/demo     # for building demos

ifneq (,$(filter-out runtest/%,$(filter-out clean%,$(MAKECMDGOALS))))
  -include $(addsuffix .d,$(subst $(EXE),,$(MAKECMDGOALS)))
endif

ifneq (,$(filter runtest/%,$(MAKECMDGOALS)))
  -include $(addsuffix .d,$(subst runtest/,,$(MAKECMDGOALS)))
endif

ifneq (,$(filter runtest_no_fail/%,$(MAKECMDGOALS)))
  -include $(addsuffix .d,$(subst runtest_no_fail/,,$(MAKECMDGOALS)))
endif

all: build docs
build: libstan.a stanc
docs: manual doxygen

##
# Clean up.
##
.PHONY: clean clean-demo clean-dox clean-manual clean-models clean-all
clean:
	$(RM) -r *.dSYM
	$(RM) src/test/gm/model_specs/*.cpp

clean-dox:
	$(RM) -r doc/api

clean-manual:
	cd src/docs/stan-reference; $(RM) *.aux *.bbl *.blg *.log *.toc *.pdf *.out

clean-models:
	$(RM) -r models $(MODEL_HEADER).gch $(MODEL_HEADER).pch $(MODEL_HEADER).d

clean-demo:
	$(RM) -r demo

clean-all: clean clean-models clean-dox clean-manual clean-models clean-demo
	$(RM) -r test bin doc

