STAN_HOME = ../../../../..
BOOSTPATH = $(shell find $(STAN_HOME)/lib -path '*lib/boost_*' -regex '.*lib\/boost_[^/]*')
EIGENPATH = $(shell find $(STAN_HOME)/lib -path '*lib/eigen_*' -regex '.*lib\/eigen_[^/]*')
CPPFLAGS = -I $(BOOSTPATH)  -I$(EIGENPATH) -I $(STAN_HOME)/src
LIBFLAGS = -L$(STAN_HOME)/bin -lstan 

lsat :
	$(STAN_HOME)/bin/stanc --name=lsat lsat.stan 
	g++ -O0  -DNDEBUG $(CPPFLAGS) lsat.cpp -o lsat $(LIBFLAGS)
	./lsat --data=lsat.data.R

clean :
	rm -f lsat.cpp samples.csv lsat

