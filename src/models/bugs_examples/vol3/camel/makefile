STAN_HOME = ../../../../..
PGM = camel#
PGM2 = camel2#
CXX = g++ 
BOOSTPATH = $(shell find $(STAN_HOME)/lib -path '*lib/boost_*' -regex '.*lib\/boost_[^/]*')
EIGENPATH = $(shell find $(STAN_HOME)/lib -path '*lib/eigen_*' -regex '.*lib\/eigen_[^/]*')
CPPFLAGS = -I $(BOOSTPATH)  -I$(EIGENPATH) -I $(STAN_HOME)/src
LIBFLAGS = -L$(STAN_HOME)/bin -lstan 

$(PGM) :
	$(STAN_HOME)/bin/stanc  --name=$(PGM)  $(PGM).stan 
	$(CXX) -O3 -DNDEBUG $(CPPFLAGS)  $(PGM).cpp -o $(PGM) $(LIBFLAGS) 
	./$(PGM) # --data=$(PGM).data.R

$(PGM2) :
	$(STAN_HOME)/bin/stanc  --name=$(PGM2)  $(PGM2).stan 
	$(CXX) -O3 -DNDEBUG $(CPPFLAGS)  $(PGM2).cpp -o $(PGM2) $(LIBFLAGS) 
	./$(PGM2) # --data=$(PGM2).data.R

clean :
	rm -f $(PGM).cpp samples.csv $(PGM) 
	rm -f $(PGM2).cpp samples.csv $(PGM2) 
