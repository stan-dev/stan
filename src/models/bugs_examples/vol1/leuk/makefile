STAN_HOME = ../../../../..
PGM = leuk

$(PGM) :
	$(STAN_HOME)/bin/stanc  --name=$(PGM)  $(PGM).stan.0 
	g++   -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) --data=$(PGM).Rdata # --iters=2000

clean :
	rm $(PGM).cpp samples.csv $(PGM) 
