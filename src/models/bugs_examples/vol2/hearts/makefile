STAN_HOME = ../../../../..
PGM = hearts#
CC = g++-4.6 -std=c++0x 

$(PGM) :
	cat $(PGM).stan | $(STAN_HOME)/demo/gm > $(PGM).cpp
	$(CC) -O0 -DNDEBUG  -std=c++0x -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) --data_file=$(PGM).Rdata

clean :
	rm $(PGM).cpp samples.csv $(PGM) 
