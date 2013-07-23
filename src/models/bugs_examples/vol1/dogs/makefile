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
	./$(PGM) --data=$(PGM).data.R --seed=1340384924 --chain_id=4 --diagnostics=d.csv# --iters=20000

clean :
	rm -rf $(PGM).cpp samples.csv $(PGM) 
