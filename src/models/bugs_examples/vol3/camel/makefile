STAN_HOME = ../../../../..
PGM = camel#
PGM2 = camel2#
CC = g++ 
CC = clang++ 

$(PGM) :
	$(STAN_HOME)/bin/stanc  --name=$(PGM)  $(PGM).stan 
	$(CC) -O0 -DNDEBUG -g -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) # --data=$(PGM).Rdata

$(PGM2) :
	$(STAN_HOME)/bin/stanc  --name=$(PGM2)  $(PGM2).stan 
	$(CC) -O0 -DNDEBUG -g -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM2).cpp -o $(PGM2)
	./$(PGM2) # --data=$(PGM2).Rdata

clean :
	rm -f $(PGM).cpp samples.csv $(PGM) 
	rm -f $(PGM2).cpp samples.csv $(PGM2) 
