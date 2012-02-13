STAN_HOME = ../../../../..
PGM = birats
CXX = g++  -std=c++0x 

$(PGM) :
	cat $(PGM).stan | $(STAN_HOME)/demo/gm > $(PGM).cpp
	$(CXX) -O0 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) --data=$(PGM).Rdata --iter=10000 

clean :
	rm -f $(PGM).cpp samples.csv $(PGM) 
