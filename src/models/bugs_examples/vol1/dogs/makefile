STAN_HOME = ../../../../..
PGM = dogs#
CXX = clang++ 
BOOSTPATH = $(shell find $(STAN_HOME)/lib -path '*lib/boost_*' -regex '.*lib\/boost_[^/]*')
EIGENPATH = $(shell find $(STAN_HOME)/lib -path '*lib/eigen_*' -regex '.*lib\/eigen_[^/]*')
CPPFLAGS = -I $(BOOSTPATH)  -I$(EIGENPATH) -I $(STAN_HOME)/src
LIBFLAGS = -L$(STAN_HOME)/bin -lstan 

$(PGM) :
	$(STAN_HOME)/bin/stanc  --name=$(PGM)  $(PGM).stan 
	$(CXX) -Wall -O3 -DNDEBUG $(CPPFLAGS) $(PGM).cpp -o $(PGM) $(LIBFLAGS)
	./$(PGM) id=4 data=$(PGM).data.R  method=sample  num_warmup=1000 num_samples=1000 random seed=1340384924 output sample=samples.csv

clean :
	rm -rf $(PGM).cpp samples.csv $(PGM) 
