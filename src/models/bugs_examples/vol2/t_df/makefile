STAN_HOME = ../../../../..
PGM = simt4#
PGM2 = estdof#
PGM3 = estdof2#
CXX = g++  
CXX = clang++ 

$(PGM) : 
	$(STAN_HOME)/bin/stanc --name=$(PGM)  $(PGM).stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) --data=$(PGM).Rdata --samples=samples1.csv 

$(PGM2) : 
	$(STAN_HOME)/bin/stanc --name=$(PGM2)  $(PGM2).stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM2).cpp -o $(PGM2)
	./$(PGM2) --data=$(PGM2).Rdata --samples=samples2.csv 

$(PGM3) : 
	$(STAN_HOME)/bin/stanc --name=$(PGM3)  $(PGM3).stan.0 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM3).cpp -o $(PGM3)
	./$(PGM3) --data=$(PGM3).Rdata --samples=samples2.csv 

clean :
	rm -f $(PGM).cpp samples1.csv $(PGM) 
	rm -f $(PGM2).cpp samples2.csv $(PGM2) 
	rm -f $(PGM3).cpp samples3.csv $(PGM3) 
