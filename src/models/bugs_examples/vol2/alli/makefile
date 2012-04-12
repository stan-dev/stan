STAN_HOME = ../../../../..
PGM = alli#
CXX = g++  
CXX = clang++ 

.PHONY : one two 


two : 
	$(STAN_HOME)/bin/stanc --name=$(PGM)2  $(PGM)2.stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)2.cpp -o $(PGM)2
	./$(PGM)2 --data=$(PGM)2.Rdata --samples=samples.csv 

one : 
	$(STAN_HOME)/bin/stanc --name=$(PGM)1  $(PGM)1.stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)1.cpp -o $(PGM)1
	./$(PGM)1 --data=$(PGM).Rdata --samples=samples1.csv 

.PHONY : clean 

clean :
	rm $(PGM)1.cpp samples1.csv $(PGM)1 
	rm $(PGM)2.cpp samples2.csv $(PGM)2
