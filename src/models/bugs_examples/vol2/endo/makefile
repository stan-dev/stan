STAN_HOME = ../../../../..
PGM = endo#
CXX = clang++ -std=c++0x 
CXX = g++-4.6.2 -std=c++0x 

.PHONY : one two three 

one : 
	cat $(PGM)1.stan | $(STAN_HOME)/demo/gm > $(PGM)1.cpp
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)1.cpp -o $(PGM)1
	./$(PGM)1 --data_file=$(PGM).Rdata --sample_file=samples1.csv 

two : 
	cat $(PGM)2.stan | $(STAN_HOME)/demo/gm > $(PGM)2.cpp
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)2.cpp -o $(PGM)2
	./$(PGM)2 --data_file=$(PGM).Rdata --sample_file=samples2.csv 

three : 
	cat $(PGM)3.stan | $(STAN_HOME)/demo/gm > $(PGM)3.cpp
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)3.cpp -o $(PGM)3
	./$(PGM)3 --data_file=$(PGM)2.Rdata --sample_file=samples3.csv 

.PHONY : clean 

clean :
	rm $(PGM)1.cpp samples1.csv $(PGM)1 
	rm $(PGM)2.cpp samples2.csv $(PGM)2
	rm $(PGM)3.cpp samples3.csv $(PGM)3
