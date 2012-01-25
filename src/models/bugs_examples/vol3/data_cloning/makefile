STAN_HOME = ../../../../..
PGM = seeds#
CXX = clang++ 

$(PGM) :
	cat $(PGM).stan | $(STAN_HOME)/demo/gm > $(PGM).cpp
	$(CXX) -O0 -DNDEBUG  -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) --data=$(PGM).Rdata

clean :
	rm -rf $(PGM).cpp samples.csv $(PGM) 
