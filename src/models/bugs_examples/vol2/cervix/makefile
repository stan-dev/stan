STAN_HOME = ../../../../..
PGM = cervix#
CC = g++  -std=c++0x 

$(PGM) :
	cat $(PGM).stan | $(STAN_HOME)/demo/gm > $(PGM).cpp
	$(CC) -O3 -DNDEBUG  -std=c++0x -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) --data=$(PGM).Rdata

clean :
	rm $(PGM).cpp samples.csv $(PGM) 
