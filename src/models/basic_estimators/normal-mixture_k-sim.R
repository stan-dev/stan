library("rstan");
source("normal-mixture_k.data.R")
fit <- stan(file="normal_mixture_k.stan", data=c("K","N","y"), iter=1000, chains=1, init=0);
