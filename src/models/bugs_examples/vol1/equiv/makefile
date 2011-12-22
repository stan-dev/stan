STAN_HOME = ../../../../..
PGM = equiv#
CXX = g++ -std=c++0x 

$(PGM) :
	cat	$(PGM).stan | $(STAN_HOME)/demo/gm > $(PGM).cpp
	$(CXX) -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) --data_file=$(PGM).Rdata --num_burnin=1000 --num_iterations=10000

clean :
	rm $(PGM).cpp samples.csv $(PGM) 
