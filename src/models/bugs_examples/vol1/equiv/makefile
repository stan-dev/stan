STAN_HOME = ../../../../..
PGM = equiv#
CXX = g++  

$(PGM) :
	$(STAN_HOME)/bin/stanc  --name=$(PGM) $(PGM).stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) --data=$(PGM).Rdata --burnin=1000 --iters=10000

clean :
	rm $(PGM).cpp samples.csv $(PGM) 
