STAN_HOME = ../../../../..
PGM = inhalers#
CXX = g++  
CXX = clang++ 

$(PGM) :
	$(STAN_HOME)/bin/stanc  --name=$(PGM)  $(PGM).stan.0 
	$(CXX) -g -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) --data=$(PGM).Rdata --iter=2000

clean :
	rm -f $(PGM).cpp samples.csv $(PGM) 
