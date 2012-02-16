STAN_HOME = ../../../../..
PGM = inhalers#
CXX = g++ -std=gnu++0x 
CXX = clang++ 

$(PGM) :
	cat $(PGM).stan.0 | $(STAN_HOME)/demo/gm > $(PGM).cpp
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) --data=$(PGM).Rdata --iter=2000

clean :
	rm -f $(PGM).cpp samples.csv $(PGM) 
