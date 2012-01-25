STAN_HOME = ../../../../..
PGM = kidney


$(PGM) :
	cat $(PGM).stan | $(STAN_HOME)/demo/gm > $(PGM).cpp
	clang++ -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) --data=$(PGM)2.Rdata # --iter=2000

clean :
	rm $(PGM).cpp samples.csv $(PGM) 
