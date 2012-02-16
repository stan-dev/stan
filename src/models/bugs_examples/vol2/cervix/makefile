STAN_HOME = ../../../../..
PGM = cervix#
PGM = cervix2#
CC = g++  -std=c++0x 

$(PGM) :
	cat $(PGM).stan | $(STAN_HOME)/demo/gm > $(PGM).cpp
	$(CC) -O3 -DNDEBUG  -std=c++0x -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) --data=$(PGM).Rdata --iter=10000 --thin=1

clean :
	rm $(PGM).cpp samples.csv $(PGM) 
