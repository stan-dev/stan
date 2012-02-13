STAN_HOME = ../../../../..
PGM = dyes#
CXX = g++ -std=c++0x 
CXX = clang++ 

$(PGM) :
	cat $(PGM).stan | $(STAN_HOME)/demo/gm > $(PGM).cpp
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) --data=$(PGM).Rdata --iter=100000

clean :
	rm $(PGM).cpp samples.csv $(PGM) 
