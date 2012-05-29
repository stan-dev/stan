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

##
# Set default compiler options.
## 
CFLAGS = -I src -I lib -O$O -Wall
LDLIBS = -Lbin -lstan
LDLIBS_STANC = -Lbin -lstanc
EXE = 

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

##
# Get information about the compiler used.
# - CC_TYPE: {g++, clang++, mingw32-g++, other}
# - CC_MAJOR: major version of CC
# - CC_MINOR: minor version of CC
##
-include make/detect_cc
# FIXME: verify compiler


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


.PHONY: help
help:
	@echo '------------------------------------------------------------'
	@echo 'Stan: makefile'
	@echo '  Current configuration:'
	@echo '  - OS (Operating System):   ' $(OS)
	@echo '  - CC (Compiler):           ' $(CC)
	@echo '  - O (Optimize Level):      ' $(O)
#	@echo '  - EXE (Executable posfix): ' $(EXE)
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



##
# Clean up.
##
.PHONY: clean clean-demo clean-dox clean-manual clean-models clean-all
clean:
	$(RM) -r *.dSYM
	$(RM) $(LIBSTAN_OFILES) bin/libstan.a

clean-dox:
	$(RM) -r doc/api

clean-manual:
	cd src/docs/stan-reference; $(RM) *.aux *.bbl *.blg *.log *.toc *.pdf
	$(RM) doc/stan-reference.pdf

clean-models:
	$(RM) -r models $(MODEL_HEADER).gch $(MODEL_HEADER).pch


clean-demo:
	$(RM) -r demo



clean-all: clean clean-models clean-dox clean-demo
	$(RM) -r test bin doc

