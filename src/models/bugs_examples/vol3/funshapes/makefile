STAN_HOME = ../../../../..
PGM = circle#
PGM2 = squaremc#
PGM3 = ring#
PGM4 = hsquare#
PGM5 = parallelagram#

CXX = g++
CXX = clang++
BOOSTPATH = $(shell find $(STAN_HOME)/lib -path '*lib/boost_*' -regex '.*lib\/boost_[^/]*')
EIGENPATH = $(shell find $(STAN_HOME)/lib -path '*lib/eigen_*' -regex '.*lib\/eigen_[^/]*')
CPPFLAGS = -I $(BOOSTPATH)  -I$(EIGENPATH) -I $(STAN_HOME)/src
LIBFLAGS = -L$(STAN_HOME)/bin -lstan 

$(PGM) :
	$(STAN_HOME)/bin/stanc --name=$(PGM)  $(PGM).stan 
	$(CXX) -Wall -O3 -DNDEBUG $(CPPFLAGS) $(PGM).cpp -o $(PGM) $(LIBFLAGS)
	./$(PGM) # --data=$(PGM).data.R

$(PGM2) :
	$(STAN_HOME)/bin/stanc --name=$(PGM2)  $(PGM2).stan
	$(CXX) -Wall -O3 -DNDEBUG $(CPPFLAGS) $(PGM2).cpp -o $(PGM2) $(LIBFLAGS)
	./$(PGM2) --iter=10000 --warmup=1000  # --data=$(PGM2).data.R


$(PGM3) :
	$(STAN_HOME)/bin/stanc --name=$(PGM3)  $(PGM3).stan
	$(CXX) -Wall -O3 -DNDEBUG $(CPPFLAGS) $(PGM3).cpp -o $(PGM3) $(LIBFLAGS)
	./$(PGM3) --iter=10000 --warmup=1000 --thin=1# --data=$(PGM3).data.R

$(PGM4) :
	$(STAN_HOME)/bin/stanc --name=$(PGM4)  $(PGM4).stan.0 
	$(CXX) -Wall -O3 -DNDEBUG $(CPPFLAGS) $(PGM4).cpp -o $(PGM4) $(LIBFLAGS)
	./$(PGM4) # --data=$(PGM4).data.R

$(PGM5) :
	$(STAN_HOME)/bin/stanc --name=$(PGM5)  $(PGM5).stan
	$(CXX) -Wall -O3 -DNDEBUG $(CPPFLAGS) $(PGM5).cpp -o $(PGM5) $(LIBFLAGS)
	./$(PGM5) # --data=$(PGM5).data.R


clean :
	rm -f $(PGM).cpp samples.csv $(PGM) 
	rm -f $(PGM2).cpp samples.csv $(PGM2) 
	rm -f $(PGM3).cpp samples.csv $(PGM3) 
	rm -f $(PGM4).cpp samples.csv $(PGM4) 
	rm -f $(PGM5).cpp samples.csv $(PGM5) 
