# To reproduce the segfault problem due to the compiled code
# had already been deleted before some code in the DSO 
# gets called (This segfault cannot be reproduced by this 
# code though.)
# 
library(rstan)
# example(stanc)


stanmodelcode <- '
data {
  int<lower=0> N;
  real y[N];
} 

parameters {
  real mu;
} 

model {
  mu ~ normal(0, 10);
  y ~ normal(mu, 1); 
} 

'
model_name <- "normal1"; 

rr <- stan_model(model_code = stanmodelcode, model_name = model_name, 
                 verbose = TRUE) 

y <- rnorm(20) 
mean(y) 
sd(y)
dat <- list(N = 20L, y = y) 

mod <- rr@dso@.CXXDSOMISC$module 
model_cppname <- rr@model_cpp$model_cppname 
stan_fit_cpp_module <- eval(call("$", mod, paste('stan_fit4', model_cppname, sep = '')))
sampler <- new(stan_fit_cpp_module, dat, function() {NULL})
rm(stan_fit_cpp_module)
gc()
args <- list(init = list(mu = 2))
s <- try(sampler$call_sampler(args))

rm(rr)
getLoadedDLLs()
gc()
getLoadedDLLs()
s2 <- try(sampler$call_sampler(args))


crtsf <- function() { 
  fit <- stan(model_code = stanmodelcode, data = dat)
  fit 
} 

fit <- crtsf()
gc()
getLoadedDLLs()
s3 <- try(sampler$call_sampler(args))
