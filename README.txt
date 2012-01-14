STAN README
========================================
 
STAN is a C++ library for probability, optimization and sampling.

 
Licesning
------------------- 
STAN is licensed under the new BSD license.  See
doc/LICENSE.txt

 
Header-Only Package
------------------- 
It is distributed as a header-only lib, so there's nothing 
to install.  Just unpack into a directory we'll call $STAN.
 
External Dependencies
------------------- 
These are listed in doc/DEPEDENCIES.txt.  These should be 
resolved before continuing. Note that if you are compiling
with clang++ on Linux, make sure have the development files
installed, which on Debian are in the libclang-dev package.

 
Unit Tests
------------------- 
There is a top-level makefile. 
 
% cd $STAN 
% make test-all

 
Demos
------------------- 
Coming soon.
 
