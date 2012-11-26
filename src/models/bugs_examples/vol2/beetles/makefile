STAN_HOME = ../../../../..
BOOSTPATH = $(shell find $(STAN_HOME)/lib -path '*lib/boost_*' -regex '.*lib\/boost_[^/]*')
EIGENPATH = $(shell find $(STAN_HOME)/lib -path '*lib/eigen_*' -regex '.*lib\/eigen_[^/]*')
CPPFLAGS = -I $(BOOSTPATH)  -I$(EIGENPATH) -I $(STAN_HOME)/src
LIBFLAGS = -L$(STAN_HOME)/bin -lstan 

beetles_logit :
	$(STAN_HOME)/bin/stanc --name=beetles_logit  beetles_logit.stan 
	g++ -O0 -DNDEBUG $(CPPFLAGS) beetles_logit.cpp -o beetles_logit $(LIBFLAGS)
	./beetles_logit --data=beetles_logit.data.R

beetles_probit :
	$(STAN_HOME)/bin/stanc --name=beetles_probit  beetles_probit.stan 
	g++ -O0 -DNDEBUG $(CPPFLAGS)  beetles_probit.cpp -o beetles_probit $(LIBFLAGS)   
	./beetles_probit --data=beetles_probit.data.R

beetles_cloglog :
	$(STAN_HOME)/bin/stanc --name=beetles_cloglog  beetles_cloglog.stan 
	g++ -O0 -DNDEBUG $(CPPFLAGS) beetles_cloglog.cpp -o beetles_cloglog $(LIBFLAGS)
	./beetles_cloglog --data=beetles_cloglog.data.R

clean :
	rm -f beetles*.cpp samples.csv beetles_logit beetles_probit beetles_cloglog
