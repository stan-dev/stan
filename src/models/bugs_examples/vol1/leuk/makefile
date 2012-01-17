STAN_HOME = ../../../../..
PGM = leuk

$(PGM) :
	cat $(PGM).stan | $(STAN_HOME)/demo/gm > $(PGM).cpp
	g++ -std=c++0x  -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) --data=$(PGM).Rdata # --iters=2000

clean :
	rm $(PGM).cpp samples.csv $(PGM) 
