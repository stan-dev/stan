STAN_HOME = ../../../../..
PGM = stacks#
CXX = g++ -std=c++0x 

normal :
	cat $(PGM)_normal.stan | $(STAN_HOME)/demo/gm > $(PGM)_normal.cpp
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)_normal.cpp -o $(PGM)_normal 
	./$(PGM)_normal --data_file=$(PGM)2.Rdata  --sample_file=samples_normal.csv 

normal2 :
	cat $(PGM)_normal2.stan | $(STAN_HOME)/demo/gm > $(PGM)_normal2.cpp
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)_normal2.cpp -o $(PGM)_normal2 
	./$(PGM)_normal2 --data_file=$(PGM)2.Rdata  --sample_file=samples_normal.csv 

dexp :
	cat $(PGM)_dexp.stan | $(STAN_HOME)/demo/gm > $(PGM)_dexp.cpp
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)_dexp.cpp -o $(PGM)_dexp 
	./$(PGM)_dexp --data_file=$(PGM)2.Rdata --sample_file=samples_dexp.csv 

dexp2 :
	cat $(PGM)_dexp2.stan | $(STAN_HOME)/demo/gm > $(PGM)_dexp2.cpp
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)_dexp2.cpp -o $(PGM)_dexp2 
	./$(PGM)_dexp2 --data_file=$(PGM)2.Rdata --sample_file=samples_dexp.csv 

t :
	cat $(PGM)_t.stan | $(STAN_HOME)/demo/gm > $(PGM)_t.cpp
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)_t.cpp -o $(PGM)_t 
	./$(PGM)_t --data_file=$(PGM)2.Rdata --sample_file=samples_t.csv 

t2 :
	cat $(PGM)_t2.stan | $(STAN_HOME)/demo/gm > $(PGM)_t2.cpp
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)_t2.cpp -o $(PGM)_t2 
	./$(PGM)_t2 --data_file=$(PGM)2.Rdata --sample_file=samples_t.csv --num_iterations=10000 --num_burnin=1000 

clean :
	rm $(PGM)_normal.cpp samples_normal.csv $(PGM)_normal
	rm $(PGM)_normal2.cpp  $(PGM)_normal2
	rm $(PGM)_dexp.cpp samples_dexp.csv $(PGM)_dexp
	rm $(PGM)_dexp2.cpp  $(PGM)_dexp2
	rm $(PGM)_t.cpp samples_t.csv $(PGM)_t
	rm $(PGM)_t2.cpp  $(PGM)_t2
