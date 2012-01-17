STAN_HOME = ../../../../..
PGM = alli#
CXX = clang++ -std=c++0x 
CXX = g++-4.6.2 -std=c++0x 

.PHONY : one two 

one : 
	cat $(PGM)1.stan | $(STAN_HOME)/demo/gm > $(PGM)1.cpp
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)1.cpp -o $(PGM)1
	./$(PGM)1 --data=$(PGM).Rdata --samples=samples1.csv 

two : 
	cat $(PGM)2.stan | $(STAN_HOME)/demo/gm > $(PGM)2.cpp
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)2.cpp -o $(PGM)2
	./$(PGM)2 --data=$(PGM).Rdata --samples=samples2.csv 

.PHONY : clean 

clean :
	rm $(PGM)1.cpp samples1.csv $(PGM)1 
	rm $(PGM)2.cpp samples2.csv $(PGM)2
