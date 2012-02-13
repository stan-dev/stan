STAN_HOME = ../../../../..
PGM = mice


$(PGM) :
	cat $(PGM).stan | $(STAN_HOME)/demo/gm > $(PGM).cpp
	clang++ -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) --data=$(PGM).Rdata --iters=20000

clean :
	rm -f $(PGM).cpp samples.csv $(PGM) 
