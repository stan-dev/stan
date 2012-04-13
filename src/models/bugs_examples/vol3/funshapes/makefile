STAN_HOME = ../../../../..
PGM = circle#
PGM2 = squaremc#
PGM3 = ring#
PGM4 = hsquare#
PGM5 = parallelagram#

CXX = g++
CXX = clang++

$(PGM) :
	$(STAN_HOME)/bin/stanc --name=$(PGM)  $(PGM).stan 
	$(CXX) -O0 -DNDEBUG  -std=c++0x -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) # --data=$(PGM).Rdata

$(PGM2) :
	$(STAN_HOME)/bin/stanc --name=$(PGM2)  $(PGM2).stan.0 
	$(CXX) -O0 -DNDEBUG  -std=c++0x -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM2).cpp -o $(PGM2)
	./$(PGM2) # --data=$(PGM2).Rdata


$(PGM3) :
	$(STAN_HOME)/bin/stanc --name=$(PGM3)  $(PGM3).stan.0 
	$(CXX) -O0 -DNDEBUG  -std=c++0x -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM3).cpp -o $(PGM3)
	./$(PGM3) # --data=$(PGM3).Rdata

$(PGM4) :
	$(STAN_HOME)/bin/stanc --name=$(PGM4)  $(PGM4).stan.0 
	$(CXX) -O0 -DNDEBUG  -std=c++0x -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM4).cpp -o $(PGM4)
	./$(PGM4) # --data=$(PGM4).Rdata

$(PGM5) :
	$(STAN_HOME)/bin/stanc --name=$(PGM5)  $(PGM5).stan
	$(CXX) -O0 -DNDEBUG  -std=c++0x -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM5).cpp -o $(PGM5)
	./$(PGM5) # --data=$(PGM5).Rdata


clean :
	rm -f $(PGM).cpp samples.csv $(PGM) 
	rm -f $(PGM2).cpp samples.csv $(PGM2) 
	rm -f $(PGM3).cpp samples.csv $(PGM3) 
	rm -f $(PGM4).cpp samples.csv $(PGM4) 
	rm -f $(PGM5).cpp samples.csv $(PGM5) 
