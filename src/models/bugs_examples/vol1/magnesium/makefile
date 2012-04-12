STAN_HOME = ../../../../..
PGM = magnesium#
CXX = g++  
CXX = clang++ 

$(PGM) :
	$(STAN_HOME)/bin/stanc  --name=$(PGM)  $(PGM)_all.stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM) 
	./$(PGM) --data=$(PGM).Rdata --burnin=1000 --iters=10000 # --samples=samples.csv 

.PHONY: all 

all : one two three four five six

one : 
	$(STAN_HOME)/bin/stanc --name=$(PGM)1  $(PGM)1.stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)1.cpp -o $(PGM)1 
	./$(PGM)1 --data=$(PGM).Rdata --burnin=1000 --iters=2000 --samples=samples1.csv 

two :
	$(STAN_HOME)/bin/stanc --name=$(PGM)2  $(PGM)2.stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)2.cpp -o $(PGM)2 
	./$(PGM)2 --data=$(PGM).Rdata --burnin=1000 --iters=10000 --samples=samples2.csv 

three :
	$(STAN_HOME)/bin/stanc -name=$(PGM)3  $(PGM)3.stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)3.cpp -o $(PGM)3 
	./$(PGM)3 --data=$(PGM).Rdata --burnin=1000 --iters=10000 --samples=samples3.csv 

four : 
	$(STAN_HOME)/bin/stanc --name=$(PGM)4  $(PGM)4.stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)4.cpp -o $(PGM)4 
	./$(PGM)4 --data=$(PGM).Rdata --burnin=1000 --iters=10000 --samples=samples4.csv 

five : 
	$(STAN_HOME)/bin/stanc --name=$(PGM)5  $(PGM)5.stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)5.cpp -o $(PGM)5 
	./$(PGM)5 --data=$(PGM).Rdata --burnin=1000 --iters=10000 --samples=samples5.csv 

six : 
	$(STAN_HOME)/bin/stanc --name=$(PGM)6  $(PGM)6.stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)6.cpp -o $(PGM)6 
	./$(PGM)6 --data=$(PGM).Rdata --burnin=1000 --iters=10000 --samples=samples6.csv 

clean :
	rm -f $(PGM).cpp samples.csv $(PGM)1 
	rm -f $(PGM)1.cpp samples1.csv $(PGM)1 
	rm -f $(PGM)2.cpp samples2.csv $(PGM)2 
	rm -f $(PGM)3.cpp samples3.csv $(PGM)3 
	rm -f $(PGM)4.cpp samples4.csv $(PGM)4 
	rm -f $(PGM)5.cpp samples5.csv $(PGM)5 
	rm -f $(PGM)6.cpp samples6.csv $(PGM)6 
