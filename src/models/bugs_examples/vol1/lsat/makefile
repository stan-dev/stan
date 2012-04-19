STAN_HOME = ../../../../..

lsat :
	$(STAN_HOME)/bin/stanc --name=lsat lsat.stan 
	g++ -O0  -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src lsat.cpp -o lsat
	./lsat --data=lsat.Rdata

clean :
	rm -f lsat.cpp samples.csv lsat

