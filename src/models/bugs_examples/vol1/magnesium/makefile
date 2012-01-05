STAN_HOME = ../../../../..
PGM = magnesium#
CXX = g++ -std=c++0x 

$(PGM) :
	cat $(PGM)_all.stan | $(STAN_HOME)/demo/gm > $(PGM).cpp 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM) 
	./$(PGM) --data_file=$(PGM).Rdata --num_burnin=1000 --num_iterations=10000 # --sample_file=samples.csv 

.PHONY: all 

all : one two three four five six

one : 
	cat $(PGM)1.stan | $(STAN_HOME)/demo/gm > $(PGM)1.cpp 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)1.cpp -o $(PGM)1 
	./$(PGM)1 --data_file=$(PGM).Rdata --num_burnin=1000 --num_iterations=2000 --sample_file=samples1.csv 

two :
	cat $(PGM)2.stan | $(STAN_HOME)/demo/gm > $(PGM)2.cpp 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)2.cpp -o $(PGM)2 
	./$(PGM)2 --data_file=$(PGM).Rdata --num_burnin=1000 --num_iterations=10000 --sample_file=samples2.csv 

three :
	cat $(PGM)3.stan | $(STAN_HOME)/demo/gm > $(PGM)3.cpp 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)3.cpp -o $(PGM)3 
	./$(PGM)3 --data_file=$(PGM).Rdata --num_burnin=1000 --num_iterations=10000 --sample_file=samples3.csv 

four : 
	cat $(PGM)4.stan | $(STAN_HOME)/demo/gm > $(PGM)4.cpp 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)4.cpp -o $(PGM)4 
	./$(PGM)4 --data_file=$(PGM).Rdata --num_burnin=1000 --num_iterations=10000 --sample_file=samples4.csv 

five : 
	cat $(PGM)5.stan | $(STAN_HOME)/demo/gm > $(PGM)5.cpp 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)5.cpp -o $(PGM)5 
	./$(PGM)5 --data_file=$(PGM).Rdata --num_burnin=1000 --num_iterations=10000 --sample_file=samples5.csv 

six : 
	cat $(PGM)6.stan | $(STAN_HOME)/demo/gm > $(PGM)6.cpp 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)6.cpp -o $(PGM)6 
	./$(PGM)6 --data_file=$(PGM).Rdata --num_burnin=1000 --num_iterations=10000 --sample_file=samples6.csv 

clean :
	rm $(PGM).cpp samples.csv $(PGM)1 
	rm $(PGM)1.cpp samples1.csv $(PGM)1 
	rm $(PGM)2.cpp samples2.csv $(PGM)2 
	rm $(PGM)3.cpp samples3.csv $(PGM)3 
	rm $(PGM)4.cpp samples4.csv $(PGM)4 
	rm $(PGM)5.cpp samples5.csv $(PGM)5 
	rm $(PGM)6.cpp samples6.csv $(PGM)6 
