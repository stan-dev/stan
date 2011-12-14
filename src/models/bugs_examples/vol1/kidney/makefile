STAN_HOME = ../../../../..
PGM = kidney


$(PGM) :
	cat $(PGM).stan | $(STAN_HOME)/demo/gm > $(PGM).cpp
	clang++ -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) --data_file=$(PGM)2.Rdata # --num_iterations=2000

clean :
	rm $(PGM).cpp samples.csv $(PGM) 
