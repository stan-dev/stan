STAN_HOME = ../../../../..
PGM = magnesium#
CXX = g++  
CXX = clang++ 

$(PGM) :
	$(STAN_HOME)/bin/stanc  --name=$(PGM)  $(PGM).stan 
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM) 
	./$(PGM) --data=$(PGM).Rdata --burnin=1000 --iters=10000 # --samples=samples.csv 

.PHONY: all 

all : magnesium

clean :
	rm -f $(PGM).cpp samples.csv $(PGM)

