STAN_HOME = ../../../../..
CPPFLAGS = -I $(STAN_HOME)/lib/boost_1.51.0 -I$(STAN_HOME)/lib/eigen_3.1.1 -I$(STAN_HOME)/lib/eigen_3.1.1/unsupported  -I $(STAN_HOME)/src
LIBFLAGS = -L$(STAN_HOME)/bin -lstan 

lsat :
	$(STAN_HOME)/bin/stanc --name=lsat lsat.stan 
	g++ -O0  -DNDEBUG $(CPPFLAGS) lsat.cpp -o lsat $(LIBFLAGS)
	./lsat --data=lsat.Rdata

clean :
	rm -f lsat.cpp samples.csv lsat

