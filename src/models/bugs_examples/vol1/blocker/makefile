STAN_HOME = ../../../../..
PGM = blocker#
CXX = g++  
CPPFLAGS = -I $(STAN_HOME)/lib/boost_1.51.0 -I$(STAN_HOME)/lib/eigen_3.1.1 -I$(STAN_HOME)/lib/eigen_3.1.1/unsupported  -I $(STAN_HOME)/src

$(PGM) :
	$(STAN_HOME)/bin/stanc $(CPPFLAGS) --name=$(PGM)  $(PGM).stan 
	$(CXX) -O3 -DNDEBUG  $(PGM).cpp -o $(PGM) -L$(STAN_HOME)/bin -lstan 
	./$(PGM) --data=$(PGM).Rdata --burnin=1000 --iters=10000 

clean :
	rm -rf $(PGM).cpp samples.csv $(PGM) 
