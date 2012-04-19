STAN_HOME = ../../../../..
CXX = clang++ 

seeds :
	$(STAN_HOME)/bin/stanc --name=seeds  seeds.stan 
	$(CXX) -O0 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src seeds.cpp -o seeds
	./seeds --data=seeds.Rdata

clean :
	rm -f seeds.cpp samples.csv seeds
