STAN_HOME = ../../../../..
PGM = circle#
PGM2 = squaremc#
CC = g++

$(PGM) :
	cat $(PGM).stan | $(STAN_HOME)/demo/gm > $(PGM).cpp
	$(CC) -O0 -DNDEBUG  -std=c++0x -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) # --data_file=$(PGM).Rdata

$(PGM2) :
	cat $(PGM2).stan | $(STAN_HOME)/demo/gm > $(PGM2).cpp
	$(CC) -O0 -DNDEBUG  -std=c++0x -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM2).cpp -o $(PGM2)
	./$(PGM2) # --data_file=$(PGM2).Rdata

clean :
	rm -f $(PGM).cpp samples.csv $(PGM) 
	rm -f $(PGM2).cpp samples.csv $(PGM2) 
