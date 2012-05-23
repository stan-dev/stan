STAN_HOME = ../../../../..
PGM = bones#
CXX = clang++ 

$(PGM) :
	$(STAN_HOME)/bin/stanc  --name=$(PGM)  $(PGM).stan
	$(CXX)   -O3 -Wall -Wunused-function -g  -Wno-tautological-compare -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM) -L$(STAN_HOME)/bin -lstan 
	./$(PGM) --data=$(PGM).Rdata # --iters=2000

clean :
	rm -rf $(PGM).cpp samples.csv $(PGM) 
