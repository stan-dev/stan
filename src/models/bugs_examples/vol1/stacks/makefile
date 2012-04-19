STAN_HOME = ../../../../..
PGM = stacks#
CXX = g++  

normal :
	$(STAN_HOME)/bin/stanc --name=$(PGM)_normal  $(PGM)_normal.stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)_normal.cpp -o $(PGM)_normal 
	./$(PGM)_normal --data=$(PGM).Rdata  --samples=samples_normal.csv 

normal2 :
	$(STAN_HOME)/bin/stanc --name=$(PGM)_normal2  $(PGM)_normal2.stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)_normal2.cpp -o $(PGM)_normal2 
	./$(PGM)_normal2 --data=$(PGM).Rdata  --samples=samples_normal.csv 

dexp :
	$(STAN_HOME)/bin/stanc --name=$(PGM)_dexp  $(PGM)_dexp.stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)_dexp.cpp -o $(PGM)_dexp 
	./$(PGM)_dexp --data=$(PGM).Rdata --samples=samples_dexp.csv 

dexp2 :
	$(STAN_HOME)/bin/stanc --name=$(PGM)_dexp2  $(PGM)_dexp2.stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)_dexp2.cpp -o $(PGM)_dexp2 
	./$(PGM)_dexp2 --data=$(PGM).Rdata --samples=samples_dexp.csv 

t :
	$(STAN_HOME)/bin/stanc --name=$(PGM)_t  $(PGM)_t.stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)_t.cpp -o $(PGM)_t 
	./$(PGM)_t --data=$(PGM).Rdata --samples=samples_t.csv 

t2 :
	$(STAN_HOME)/bin/stanc --name=$(PGM)_t2  $(PGM)_t2.stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)_t2.cpp -o $(PGM)_t2 
	./$(PGM)_t2 --data=$(PGM).Rdata --samples=samples_t.csv --iters=10000 --burnin=1000 

clean :
	rm -f $(PGM)_normal.cpp samples_normal.csv $(PGM)_normal
	rm -f $(PGM)_normal2.cpp  $(PGM)_normal2
	rm -f $(PGM)_dexp.cpp samples_dexp.csv $(PGM)_dexp
	rm -f $(PGM)_dexp2.cpp  $(PGM)_dexp2
	rm -f $(PGM)_t.cpp samples_t.csv $(PGM)_t
	rm -f $(PGM)_t2.cpp  $(PGM)_t2
