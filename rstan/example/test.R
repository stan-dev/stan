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
dat <- list(N = 20, y = y); 
f <- sampling(rr, data = dat, init = 0, iter = 2012, sample_file = 'norm1.csv')

sampling(rr, data = dat, iter = 2012, chains = 1,
         init = list(list(mu = 2)), seed = 3, thin = 1, 
         sample_file = 'norm1.csv')

post <- read.csv(file = 'norm1.csv', header = TRUE, skip = 19, comment = '#') 
colMeans(post)


### test 2 ################ 
### this should not work since in the list of initial values, 
### mu does not exist. 
### 

dat <- list(N = 20L, y = y); 
f <- sampling(rr, data = dat, init = 0, iter = 2012, sample_file = 'norm1.csv')


mod <- rr@dso@.CXXDSOMISC$module 
model_cppname <- rr@model_cpp$model_cppname 
stan_fit_cpp_module <- eval(call("$", mod, paste('stan_fit4', model_cppname, sep = '')))
sampler <- new(stan_fit_cpp_module, dat) 
args <- list(init = list(mu = 2)) 
s <- sampler$call_sampler(args) 
args <- list(init = list(mu2 = 2)) 
s <- sampler$call_sampler(args) 
summary(s)


sampling(rr, data = dat, iter = 2012, chains = 1,
         init = list(list(mu2 = 2)), seed = 3, thin = 1, 
         sample_file = 'norm1.csv')

