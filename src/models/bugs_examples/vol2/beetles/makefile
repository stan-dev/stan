STAN_HOME = ../../../../..

beetles_logit :
	$(STAN_HOME)/bin/stanc --name=beetles_logit  beetles_logit.stan 
	g++ -O0 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src beetles_logit.cpp -o beetles_logit
	./beetles_logit --data=beetles_logit.Rdata

beetles_probit :
	$(STAN_HOME)/bin/stanc --name=beetles_probit  beetles_probit.stan 
	g++ -O0 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src beetles_probit.cpp -o beetles_probit
	./beetles_probit --data=beetles_probit.Rdata

beetles_cloglog :
	$(STAN_HOME)/bin/stanc --name=beetles_cloglog  beetles_cloglog.stan 
	g++ -O0 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src beetles_cloglog.cpp -o beetles_cloglog
	./beetles_cloglog --data=beetles_cloglog.Rdata

clean :
	rm -f beetles*.cpp samples.csv beetles_logit beetles_probit beetles_cloglog
