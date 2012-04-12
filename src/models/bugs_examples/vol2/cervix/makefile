STAN_HOME = ../../../../..
PGM = cervix#
PGM = cervix2#
CC = g++   

$(PGM) :
	$(STAN_HOME)/bin/stanc  --name=$(PGM)  $(PGM).stan 
	$(CC) -O3 -DNDEBUG   -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) --data=$(PGM).Rdata --iter=10000 --thin=1

clean :
	rm $(PGM).cpp samples.csv $(PGM) 
