STAN_HOME = ../../../../..
PGM = blocker#
CXX = g++  
BOOSTPATH = $(shell find $(STAN_HOME)/lib -path '*lib/boost_*' -regex '.*lib\/boost_[^/]*')
EIGENPATH = $(shell find $(STAN_HOME)/lib -path '*lib/eigen_*' -regex '.*lib\/eigen_[^/]*')
CPPFLAGS = -I $(BOOSTPATH)  -I$(EIGENPATH) -I $(STAN_HOME)/src
LIBFLAGS = -L$(STAN_HOME)/bin -lstan 

$(PGM) :
	$(STAN_HOME)/bin/stanc --name=$(PGM)  $(PGM).stan 
	$(CXX) -O3 -DNDEBUG  $(CPPFLAGS) $(PGM).cpp -o $(PGM) $(LIBFLAGS)
	./$(PGM) --data=$(PGM).data.R --burnin=1000 --iters=10000 

clean :
	rm -rf $(PGM).cpp samples.csv $(PGM) 
