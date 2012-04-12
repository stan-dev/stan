STAN_HOME = ../../../../..
PGM = mice


$(PGM) :
	$(STAN_HOME)/bin/stanc  --name=$(PGM)  $(PGM).stan 
	clang++ -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) --data=$(PGM).Rdata --iters=20000

clean :
	rm -f $(PGM).cpp samples.csv $(PGM) 
