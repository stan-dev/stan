STAN_HOME = ../../../../..
PGM = oxford
CPPFLAGS = -I $(STAN_HOME)/lib/boost_1.50.0 -I$(STAN_HOME)/lib/eigen_3.1.0 -I$(STAN_HOME)/lib/eigen_3.1.0/unsupported  -I $(STAN_HOME)/src
LIBFLAGS = -L$(STAN_HOME)/bin -lstan 
CXX = clang++ 

$(PGM) :
	$(STAN_HOME)/bin/stanc  --name=$(PGM)  $(PGM).stan 
	$(CXX) -O3 -DNDEBUG $(CPPFLAGS) $(PGM).cpp -o $(PGM) $(LIBFLAGS) 
	./$(PGM) --data=$(PGM).Rdata # --iter=2000

clean :
	rm -f $(PGM).cpp samples.csv $(PGM) 
