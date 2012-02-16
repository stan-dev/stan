STAN_HOME = ../../../../..

beetles_logit :
	cat beetles_logit.stan | $(STAN_HOME)/demo/gm > beetles_logit.cpp
	g++ -O0 -std=gnu++0x -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src beetles_logit.cpp -o beetles_logit
	./beetles_logit --data=beetles_logit.Rdata

beetles_probit :
	cat beetles_probit.stan | $(STAN_HOME)/demo/gm > beetles_probit.cpp
	g++ -O0 -std=gnu++0x -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src beetles_probit.cpp -o beetles_probit
	./beetles_probit --data=beetles_probit.Rdata

beetles_cloglog :
	cat beetles_cloglog.stan | $(STAN_HOME)/demo/gm > beetles_cloglog.cpp
	g++ -O0 -std=gnu++0x -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src beetles_cloglog.cpp -o beetles_cloglog
	./beetles_cloglog --data=beetles_cloglog.Rdata

clean :
	rm -f beetles*.cpp samples.csv beetles_logit beetles_probit beetles_cloglog
