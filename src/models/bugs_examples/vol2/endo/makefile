STAN_HOME = ../../../../..
PGM = endo#
CXX = g++ -std=c++0x 


$(PGM) : one 

all : one two three 

one : 
	cat $(PGM)1.stan | $(STAN_HOME)/demo/gm > $(PGM)1.cpp
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)1.cpp -o $(PGM)1
	./$(PGM)1 --data_file=$(PGM).Rdata

two : 
	cat $(PGM)2.stan | $(STAN_HOME)/demo/gm > $(PGM)2.cpp
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)2.cpp -o $(PGM)2
	./$(PGM)2 --data_file=$(PGM).Rdata

three : 
	cat $(PGM)3.stan | $(STAN_HOME)/demo/gm > $(PGM)3.cpp
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)3.cpp -o $(PGM)3
	./$(PGM)3 --data_file=$(PGM).Rdata

clean :
	rm $(PGM).cpp samples.csv $(PGM) 
