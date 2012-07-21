STAN_HOME = ../../../../..
CPPFLAGS = -I $(STAN_HOME)/lib/boost_1.50.0 -I$(STAN_HOME)/lib/eigen_3.1.0 -I$(STAN_HOME)/lib/eigen_3.1.0/unsupported  -I $(STAN_HOME)/src
LIBFLAGS = -L$(STAN_HOME)/bin -lstan 

beetles_logit :
	$(STAN_HOME)/bin/stanc --name=beetles_logit  beetles_logit.stan 
	g++ -O0 -DNDEBUG $(CPPFLAGS) beetles_logit.cpp -o beetles_logit $(LIBFLAGS)
	./beetles_logit --data=beetles_logit.Rdata

beetles_probit :
	$(STAN_HOME)/bin/stanc --name=beetles_probit  beetles_probit.stan 
	g++ -O0 -DNDEBUG $(CPPFLAGS)  beetles_probit.cpp -o beetles_probit $(LIBFLAGS)   
	./beetles_probit --data=beetles_probit.Rdata

beetles_cloglog :
	$(STAN_HOME)/bin/stanc --name=beetles_cloglog  beetles_cloglog.stan 
	g++ -O0 -DNDEBUG $(CPPFLAGS) beetles_cloglog.cpp -o beetles_cloglog $(LIBFLAGS)
	./beetles_cloglog --data=beetles_cloglog.Rdata

clean :
	rm -f beetles*.cpp samples.csv beetles_logit beetles_probit beetles_cloglog
