STAN_HOME = ../../../../..

salm :
	cat salm.stan | $(STAN_HOME)/demo/gm > salm.cpp
	g++ -O0 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src salm.cpp -o salm
	salm --data=salm.Rdata

clean :
	rm salm.cpp samples.csv salm
