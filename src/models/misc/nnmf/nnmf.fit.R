library("rstan");
source("nnmf.data.R")
stan.fit <- stan(file="nnmf.stan",
                 data = list(T=T, I=I, K=K, sigma=sigma, X=X),
                 chains=3, iter=2000, max_treedepth=9, init=0);
