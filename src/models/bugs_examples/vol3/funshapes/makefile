STAN_HOME = ../../../../..
PGM = circle#
PGM2 = squaremc#
PGM3 = ring#
PGM4 = hsquare#
PGM5 = parallelagram#

CXX = g++
CXX = clang++

CPPFLAGS = -I $(STAN_HOME)/lib/boost_1.51.0 -I$(STAN_HOME)/lib/eigen_3.1.1 -I$(STAN_HOME)/lib/eigen_3.1.1/unsupported  -I $(STAN_HOME)/src

$(PGM) :
	$(STAN_HOME)/bin/stanc --name=$(PGM)  $(PGM).stan 
	$(CXX) -Wall -O3 -DNDEBUG $(CPPFLAGS) $(PGM).cpp -o $(PGM) -L$(STAN_HOME)/bin -lstan 
	./$(PGM) # --data=$(PGM).Rdata

$(PGM2) :
	$(STAN_HOME)/bin/stanc --name=$(PGM2)  $(PGM2).stan
	$(CXX) -Wall -O3 -DNDEBUG $(CPPFLAGS) $(PGM2).cpp -o $(PGM2) -L$(STAN_HOME)/bin -lstan 
	./$(PGM2) --iter=10000 --warmup=1000  # --data=$(PGM2).Rdata


$(PGM3) :
	$(STAN_HOME)/bin/stanc --name=$(PGM3)  $(PGM3).stan
	$(CXX) -Wall -O3 -DNDEBUG $(CPPFLAGS) $(PGM3).cpp -o $(PGM3) -L$(STAN_HOME)/bin -lstan 
	./$(PGM3) --iter=10000 --warmup=1000 --thin=1# --data=$(PGM3).Rdata

$(PGM4) :
	$(STAN_HOME)/bin/stanc --name=$(PGM4)  $(PGM4).stan.0 
	$(CXX) -Wall -O3 -DNDEBUG $(CPPFLAGS) $(PGM4).cpp -o $(PGM4) -L$(STAN_HOME)/bin -lstan 
	./$(PGM4) # --data=$(PGM4).Rdata

$(PGM5) :
	$(STAN_HOME)/bin/stanc --name=$(PGM5)  $(PGM5).stan
	$(CXX) -Wall -O3 -DNDEBUG $(CPPFLAGS) $(PGM5).cpp -o $(PGM5) -L$(STAN_HOME)/bin -lstan 
	./$(PGM5) # --data=$(PGM5).Rdata


clean :
	rm -f $(PGM).cpp samples.csv $(PGM) 
	rm -f $(PGM2).cpp samples.csv $(PGM2) 
	rm -f $(PGM3).cpp samples.csv $(PGM3) 
	rm -f $(PGM4).cpp samples.csv $(PGM4) 
	rm -f $(PGM5).cpp samples.csv $(PGM5) 
