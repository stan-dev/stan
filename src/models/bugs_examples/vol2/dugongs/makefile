STAN_HOME = ../../../../..
PGM = dugongs
CXX = clang++
CPPFLAGS = -I $(STAN_HOME)/lib/boost_1.51.0 -I$(STAN_HOME)/lib/eigen_3.1.1 -I$(STAN_HOME)/lib/eigen_3.1.1/unsupported  -I $(STAN_HOME)/src
LIBFLAGS = -L$(STAN_HOME)/bin -lstan 

$(PGM) :
	$(STAN_HOME)/bin/stanc --name=$(PGM) $(PGM).stan 
	$(CXX) -O3 -DNDEBUG $(CPPFLAGS) $(PGM).cpp -o $(PGM) $(LIBFLAGS) 
	./$(PGM) --data=$(PGM).data.R

clean :
	rm -f $(PGM).cpp samples.csv $(PGM) 
