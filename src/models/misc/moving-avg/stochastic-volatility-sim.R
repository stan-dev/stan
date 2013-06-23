source("stochastic-volatility.data.R")
library("rstan");
start_time <- proc.time();
fit <- stan(file="stochastic-volatility.stan", data=list(T=T,y=y), iter=10000, chains=4, init=0);
elapsed_time <- proc.time() - start_time;
