STAN_HOME = ../../../../..
PGM = pineapple_ice


$(PGM) :
	cat $(PGM).stan | $(STAN_HOME)/demo/gm > $(PGM).cpp
	g++ -O3 -std=gnu++0x -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) --data=$(PGM).Rdata

clean :
	rm $(PGM).cpp samples.csv $(PGM) 
