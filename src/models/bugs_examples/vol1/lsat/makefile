	STAN_HOME = ../../../../..

lsat :
	 cat lsat.stan | $(STAN_HOME)/demo/gm > lsat.cpp
	g++ -std=c++0x -O0 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src lsat.cpp -o lsat
	./lsat --data=lsat.Rdata

clean :
	rm lsat.cpp samples.csv lsat
