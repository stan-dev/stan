STAN_HOME = ../../../../..
PGM = pump
CXX = clang++

$(PGM) :
	$(STAN_HOME)/bin/stanc  --name=$(PGM)  $(PGM).stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) --data=$(PGM).Rdata # --iters=2000

clean :
	rm -f $(PGM).cpp samples.csv $(PGM) 
