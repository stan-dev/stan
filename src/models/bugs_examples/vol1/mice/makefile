STAN_HOME = ../../../../..
PGM = mice


$(PGM) :
	cat $(PGM).stan | $(STAN_HOME)/demo/gm > $(PGM).cpp
	g++ -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)

run :
	./$(PGM) --data_file=$(PGM).Rdata --num_iterations=20000

rev : 
	g++ -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM)_rev.cpp -o $(PGM)

clean :
	rm $(PGM).cpp samples.csv $(PGM) 
