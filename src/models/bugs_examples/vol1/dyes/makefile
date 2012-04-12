STAN_HOME = ../../../../..
PGM = dyes#
CXX = g++  
CXX = clang++ 

$(PGM) :
	$(STAN_HOME)/bin/stanc  --name=$(PGM)  $(PGM).stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) --data=$(PGM).Rdata --iter=100000

clean :
	rm $(PGM).cpp samples.csv $(PGM) 
