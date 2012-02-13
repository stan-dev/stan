STAN_HOME = ../../../../..
CXX = clang++ 

seeds :
	cat seeds.stan | $(STAN_HOME)/demo/gm > seeds.cpp
	$(CXX) -O0 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src seeds.cpp -o seeds
	./seeds --data=seeds.Rdata

clean :
	rm seeds.cpp samples.csv seeds
