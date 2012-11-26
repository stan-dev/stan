STAN_HOME = ../../../../..
PGM = simt4#
PGM2 = estdof#
PGM3 = estdof2#
CXX = g++  
CXX = clang++ 
BOOSTPATH = $(shell find $(STAN_HOME)/lib -path '*lib/boost_*' -regex '.*lib\/boost_[^/]*')
EIGENPATH = $(shell find $(STAN_HOME)/lib -path '*lib/eigen_*' -regex '.*lib\/eigen_[^/]*')
CPPFLAGS = -I $(BOOSTPATH)  -I$(EIGENPATH) -I $(STAN_HOME)/src
LIBFLAGS = -L$(STAN_HOME)/bin -lstan 

$(PGM) : 
	$(STAN_HOME)/bin/stanc --name=$(PGM)  $(PGM).stan 
	$(CXX) -O3 -DNDEBUG $(CPPFLAGS) $(PGM).cpp -o $(PGM) $(LIBFLAGS) 
	./$(PGM) --data=$(PGM).data.R --samples=samples1.csv 

$(PGM2) : 
	$(STAN_HOME)/bin/stanc --name=$(PGM2)  $(PGM2).stan 
	$(CXX) -O3 -DNDEBUG $(CPPFLAGS) $(PGM2).cpp -o $(PGM2) $(LIBFLAGS) 
	./$(PGM2) --data=$(PGM2).data.R --samples=samples2.csv 

$(PGM3) : 
	$(STAN_HOME)/bin/stanc --name=$(PGM3)  $(PGM3).stan.0 
	$(CXX) -O3 -DNDEBUG $(CPPFLAGS) $(PGM3).cpp -o $(PGM3) $(LIBFLAGS) 
	./$(PGM3) --data=$(PGM3).data.R --samples=samples3.csv 

clean :
	rm -f $(PGM).cpp samples1.csv $(PGM) 
	rm -f $(PGM2).cpp samples2.csv $(PGM2) 
	rm -f $(PGM3).cpp samples3.csv $(PGM3) 
