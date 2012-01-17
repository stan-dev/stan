STAN_HOME = ../../../../..
PGM = oxford


$(PGM) :
	cat $(PGM).stan | $(STAN_HOME)/demo/gm > $(PGM).cpp
	g++ -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	# ./$(PGM) --data=$(PGM).Rdata

clean :
	rm $(PGM).cpp samples.csv $(PGM) 
