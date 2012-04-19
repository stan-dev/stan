STAN_HOME = ../../../../..
PGM = endo#
CXX = g++
CXX = clang++ 

.PHONY : one two three 

one : 
	$(STAN_HOME)/bin/stanc --name=$(PGM)1  $(PGM)1.stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)1.cpp -o $(PGM)1
	./$(PGM)1 --data=$(PGM)1.Rdata --samples=samples1.csv 

two : 
	$(STAN_HOME)/bin/stanc --name=$(PGM)2  $(PGM)2.stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)2.cpp -o $(PGM)2
	./$(PGM)2 --data=$(PGM)2.Rdata --samples=samples2.csv 

three : 
	$(STAN_HOME)/bin/stanc --name=$(PGM)3  $(PGM)3.stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)3.cpp -o $(PGM)3
	./$(PGM)3 --data=$(PGM)3.Rdata --samples=samples3.csv 

.PHONY : clean 

clean :
	rm -f $(PGM)1.cpp samples1.csv $(PGM)1 
	rm -f $(PGM)2.cpp samples2.csv $(PGM)2
	rm -f $(PGM)3.cpp samples3.csv $(PGM)3
