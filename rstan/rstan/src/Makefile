all: symbols.rds $(SHLIB) libstanLibrary 
LIBSTAN = ../inst/libstan
SHLIB = rstan.so

USERLIB = libstan.a

OBJECTS = stanc.o gm__grammars__whitespace_grammar_inst.o gm__grammars__program_grammar_inst.o gm__grammars__statement_grammar_inst.o  gm__ast_def.o gm__grammars__var_deccls_grammar_inst.o gm__grammars__expression_grammar_inst.o chains.o misc.o 

LIBSTANOBJECTS = agrad__agrad.o agrad__matrix.o math__matrix.o

EIGEN_PATH := $(shell $(R_HOME)/bin/Rscript -e "cat(system.file('include', package = 'RcppEigen'))")
# EIGEN_PATH = ../inst/include/stanlib/eigen_3.1.0

PKG_LIBS = $(shell  $(R_HOME)/bin/Rscript -e "Rcpp:::LdFlags()") 
PKG_CPPFLAGS += -I"../inst/include/stansrc"  
PKG_CPPFLAGS += -I"$(EIGEN_PATH)" 
PKG_CPPFLAGS += -I"$(EIGEN_PATH)/unsupported" 
PKG_CPPFLAGS += -I"../inst/include/stanlib/boost_1.52.0" 
PKG_CPPFLAGS += -I"../inst/include" 

# CXXFLAGS = -O2 -pipe -g $(LTO)
# CXXFLAGS = -Os -pipe $(LTO)
CXXFLAGS = -O3 $(LTO)
SHLIB_LDFLAGS = $(SHLIB_CXXLDFLAGS) 
SHLIB_LD = $(SHLIB_CXXLD)

$(USERLIB): $(LIBSTANOBJECTS)
	$(AR) qc $@ $^ 
	@if test -n "$(RANLIB)"; then $(RANLIB) $@; fi


libstanLibrary: $(USERLIB) 
	# -@if test ! -e $(LIBSTAN)$(R_ARCH); then mkdir -p $(LIBSTAN)$(R_ARCH); fi
	# cp $(USERLIB) $(LIBSTAN)$(R_ARCH)
	# rm $(USERLIB) 

# $(USERLIB): $(LIBSTANOBJECTS)
#   $(SHLIB_CXXLD) -s -o $(USERLIB) $(LIBSTANOBJECTS) $(SHLIB_CXXLDFLAGS) $(ALL_LIBS)
#   @if test -e "/usr/bin/install_name_tool"; then /usr/bin/install_name_tool -id $(R_PACKAGE_DIR)/libstan/$(R_ARCH)/$(USERLIB) $(USERLIB); fi

include $(R_SHARE_DIR)/make/shlib.mk

.PHONY: all clean libstanLibrary

clean: 
	rm -f $(OBJECTS) $(SHLIB) $(USERLIB) $(LIBSTANOBJECTS) 

# P.S. While Makefile is used, only one platform binary is installed 
# for example on Mac and Windows. 
