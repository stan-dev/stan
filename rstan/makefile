
# Set OS if there is no default:
uname_S := $(shell sh -c 'uname -s 2>/dev/null || echo not')
uname_O := $(shell sh -c 'uname -o 2>/dev/null || echo not')
OS = win
ifeq ($(uname_S), Darwin)
  OS = mac
endif
ifeq ($(uname_O), Cygwin)
  OS = win
endif
ifeq ($(uname_S), MINGW32_NT-6.1)
  OS = win
endif
ifeq ($(uname_S), Linux)
  OS = linux
endif

ifdef R_HOME
  R = $(R_HOME)/bin/R
  RSCRIPT = $(R_HOME)/bin/Rscript
else
  # R = R --arch x86_64
  R = R
  RSCRIPT = Rscript
endif

ifeq ($(OS), win)
  LINKCMD1 = @cmd /c mklink /D ".^\rstan^\inst^\include^\stanlib"  "..^\..^\..^\..^\lib"
  LINKCMD2 = @cmd /c mklink /D ".^\rstan^\inst^\include^\stansrc"  "..^\..^\..^\..^\src"
else
  LINKCMD1 = ln -s ../../../../lib ./rstan/inst/include/stanlib
  LINKCMD1b = ln -s ../../../$(TMP_STANLIB)/ ./rstan/inst/include/stanlib
  LINKCMD2 = ln -s ../../../../src ./rstan/inst/include/stansrc
endif
  
RSTANVER := $(shell $(RSCRIPT) -e "cat(read.dcf(file = './rstan/DESCRIPTION')[1, \"Version\"])")
STANPKG := rstan_$(RSTANVER).tar.gz 

BOOSTPATH := $(shell find ../lib -path '*lib/boost_*' -regex '.*lib\/boost_[^/]*')
EIGENPATH = $(shell find ../lib -path '*lib/eigen_*' -regex '.*lib\/eigen_[^/]*')
TMP_STANLIB := tmpstanlib#
BOOST_FOR_RSTAN := $(TMP_STANLIB)/$(subst ../lib/,,$(BOOSTPATH))
EIGEN_FOR_RSTAN := $(TMP_STANLIB)/$(subst ../lib/,,$(EIGENPATH))

# a tool in boost 
BCP = $(strip $(shell which bcp))

build $(STANPKG): ./rstan/DESCRIPTION 
	@rm -rf "./rstan/inst/include/stanlib"
	@rm -rf "./rstan/inst/include/stansrc"
ifeq ($(OS), win)
	$(LINKCMD1)
else 
	if test -d $(TMP_STANLIB); then $(LINKCMD1b); else $(LINKCMD1); fi
endif
	$(LINKCMD2)
	@rm -f $(STANPKG) 
	@echo "git_head <- function() \"`git rev-parse HEAD`\"" > rstan/R/git_head.R
	$(R) CMD build rstan --md5  # --no-vignettes


check: $(STANPKG) 
	$(R) CMD check --as-cran $(STANPKG) 

install: $(STANPKG) 
	# $(R) CMD INSTALL --no-multiarch $(STANPKG) 
	$(R) CMD INSTALL --merge-multiarch $(STANPKG) 

uninstall: $(STANPKG) 
	$(R) CMD REMOVE rstan

$(TMP_STANLIB): 
ifneq "$(BCP)" ""
	mkdir -p ${BOOST_FOR_RSTAN} 
	mkdir -p ${EIGEN_FOR_RSTAN} 
	cp -r $(EIGENPATH)/* $(EIGEN_FOR_RSTAN)
	find ../src/ -name \*\.\[ch]pp -exec bcp --scan --boost=${BOOSTPATH} '{}' ${BOOST_FOR_RSTAN} \; &> bcp.log  
	find ./rstan/inst/include -name \*\.\[ch]pp -exec bcp --scan --boost=${BOOSTPATH} '{}' ${BOOST_FOR_RSTAN} \; >> bcp.log  2>&1
else 
	@echo "cannot find boost bcp"
endif 


rmtmpstanlib: 
	rm -rf $(TMP_STANLIB) 
	rm -f bcp.log 

clean: 
	rm -f $(STANPKG) 

clean-all: 
	rm -f $(STANPKG); rm -rf $(TMP_STANLIB); rm -f bcp.log 

# buildbin:   # build a binary version  
# R CMD INSTALL -l ./tmp --build rstan

.PHONY: build check install clean clean-all
