STAN_HOME = ../../../../..
PGM = birats
CXX = g++-4.6.2 -std=c++0x 

$(PGM) :
	cat $(PGM).stan | $(STAN_HOME)/demo/gm > $(PGM).cpp
	$(CXX) -O0 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) --data_file=$(PGM).Rdata

clean :
	rm $(PGM).cpp samples.csv $(PGM) 
