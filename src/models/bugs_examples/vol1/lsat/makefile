STAN_HOME = ../../../../..

lsat :
	cat lsat.stan | $(STAN_HOME)/demo/gm > lsat.cpp
	g++ -O0 -std=gnu++0x -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src lsat.cpp -o lsat
	./lsat --data=lsat.Rdata

clean :
	rm -f lsat.cpp samples.csv lsat

