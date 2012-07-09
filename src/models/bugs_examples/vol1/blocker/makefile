STAN_HOME = ../../../../..
PGM = blocker#
CXX = g++  

$(PGM) :
	$(STAN_HOME)/bin/stanc  --name=$(PGM)  $(PGM).stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM) -L$(STAN_HOME)/bin -lstan 
	./$(PGM) --data=$(PGM).Rdata --burnin=1000 --iters=10000 

clean :
	rm -rf $(PGM).cpp samples.csv $(PGM) 
