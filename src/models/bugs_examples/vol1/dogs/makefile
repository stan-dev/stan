STAN_HOME = ../../../../..
PGM = dogs#
CXX = clang++ 
CPPFLAGS = -I $(STAN_HOME)/lib/boost_1.51.0 -I$(STAN_HOME)/lib/eigen_3.1.1 -I$(STAN_HOME)/lib/eigen_3.1.1/unsupported  -I $(STAN_HOME)/src

$(PGM) :
	$(STAN_HOME)/bin/stanc  --name=$(PGM)  $(PGM).stan 
	$(CXX) -Wall -O3 -DNDEBUG $(CPPFLAGS) $(PGM).cpp -o $(PGM) -L$(STAN_HOME)/bin -lstan 
	./$(PGM) --data=$(PGM).Rdata --seed=1340384924 --chain_id=4 # --iters=20000

clean :
	rm $(PGM).cpp samples.csv $(PGM) 
