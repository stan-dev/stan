	STAN_HOME = ../../../../..

beetles :
	cat beetles.stan | $(STAN_HOME)/demo/gm > beetles.cpp
	g++ -O0 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src beetles.cpp -o beetles
	./beetles --data_file=beetles.Rdata

clean :
	rm beetles.cpp samples.csv beetles
