STAN_HOME = ../../../../..
PGM = leukfr

$(PGM) :
	# cat $(PGM).stan | $(STAN_HOME)/demo/gm > $(PGM).cpp
	g++ -O3 -DNDEBUG -I $(STAN_HOME)/lib -I $(STAN_HOME)/src $(PGM).cpp -o $(PGM)
	./$(PGM) --data_file=$(PGM).Rdata # --num_iterations=2000

clean :
	rm $(PGM).cpp samples.csv $(PGM) 
