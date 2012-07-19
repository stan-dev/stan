
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

ifeq ($(OS), win)
  LINKCMD1 = @cmd /c mklink /J ".^\rstan^\inst^\include^\stanlib"  "..^\lib"
  LINKCMD2 = @cmd /c mklink /J ".^\rstan^\inst^\include^\stansrc"  "..^\src"
else
  LINKCMD1 = ln -s ../../../../lib ./rstan/inst/include/stanlib
  LINKCMD2 = ln -s ../../../../src ./rstan/inst/include/stansrc
endif
  
RSTANVER := $(shell Rscript -e "cat(read.dcf(file = './rstan/DESCRIPTION')[1, \"Version\"])")
STANPKG := rstan_$(RSTANVER).tar.gz 

build $(STANPKG): ./rstan/DESCRIPTION 
	@rm -rf "./rstan/inst/include/stanlib"
	@rm -rf "./rstan/inst/include/stansrc"
	$(LINKCMD1)
	$(LINKCMD2)
	@rm -f $(STANPKG) 
	R CMD build rstan

check: $(STANPKG) 
	R CMD check --as-cran $(STANPKG) 

install: $(STANPKG) 
	# R CMD INSTALL --no-multiarch $(STANPKG) 
	R CMD INSTALL $(STANPKG) 

clean: 
	rm -f $(STANPKG) 

.PHONY: build check install clean 
