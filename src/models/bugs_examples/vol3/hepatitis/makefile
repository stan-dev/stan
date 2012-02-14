STAN_HOME = ../../../../..
PGM = hepatitis#
PGM = hepatitisME#
CC = g++ -std=c++0x 

$(PGM) :
	cat $(PGM).stan | $(STAN_HOME)/demo/gm > $(PGM).cpp
	$(CC) -O0 -DNDEBUG  -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM)  --data=$(PGM).Rdata

clean :
	rm -f $(PGM).cpp samples.csv $(PGM) 
