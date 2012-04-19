STAN_HOME = ../../../../..
PGM = salm2#
CXX = clang++ 

$(PGM) : 
	$(STAN_HOME)/bin/stanc  --name=$(PGM)  $(PGM).stan 
	$(CXX) -O0 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) --data=$(PGM).Rdata

clean :
	rm -f $(PGM).cpp samples.csv $(PGM)
