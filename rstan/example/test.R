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
s <- try(sampler$call_sampler(args)) 


args <- list(init = list(mu2 = 2)) 
s <- try(sampler$call_sampler(args))
if (is(s, "try-error"))  
  message("call call_sampler error as expected")
summary(s)

fit <- sampling(rr, data = dat, iter = 2012, chains = 1,
                init = list(list(mu2 = 2)), seed = 3, thin = 1, 
                sample_file = 'norm1.csv')

print(fit)



# test fitting multiple stan model in one R session 
library(rstan)
code1 <- '
  data {
    int N; 
  } 
  parameters {
    real y[N]; 
  } 
  model {
    y ~ normal(0, 1);
  }
' 

code2 <- '
  data {
    int J; 
  } 
  parameters {
    real<lower=0, upper=1> y[J]; 
    real y2[J, J]; 
  } 
  model {
    y ~ uniform(0, 1); 
    for (j in 1:J) y2[j] ~ normal(0, 1); 
  }
'

fit1 <- stan(model_code = code1, data = list(N = 3))
fit2 <- stan(model_code = code2, data = list(J = 3))
fit2b <- stan(fit = fit2, data = list(J = 3))

## fit1b is supposed to be empty and error message pops up
## due to the wrong data fed 
fit1b <- stan(fit = fit1, data = list(J = 3))
print(fit1b) 


## test specifying pars 
fit3 <- stan(model_code = code2, data = list(J = 3))


if (!identical(fit3@model_pars, c("y", "y2", "lp__"))) 
  message("model_pars in fit3 is not as expected") 

dim3 <- list(y = 3L, y2 = c(3L, 3L), lp__ = integer(0))
if (!identical(fit3@par_dims, dim3)) 
  message("par_dims in fit3 is not as expected") 

J <- 4  
set.seed(12345)
fit4 <- stan(fit = fit3, pars = c("y2", "lp__"), data = list(J = J), seed = 8765) 
set.seed(12345)
fit5 <- stan(fit = fit3, pars = "y2", data = list(J = J), seed = get_seed(fit4)) 
if (!identical(fit4@sim, fit5@sim)) 
  message('fit4@sim is not identical with fit5@sim as supposed') 

fit4par_fnames_oi <- character(0) 
for (i in 1:J) for (j in 1:J) 
  fit4par_fnames_oi <- c(fit4par_fnames_oi, paste("y2[", j, ",", i, "]", sep = ''))
fit4par_fnames_oi <- c(fit4par_fnames_oi, "lp__")

if (!identical(fit4@sim$fnames_oi, fit4par_fnames_oi) ||  
    !identical(fit4@sim$fnames_oi, names(fit4@sim$samples[[1]]))) 
  message('fnames_oi for fit4@sim is not as expected')

if (!identical(fit4@sim$pars_oi, c('y2', 'lp__'))) 
  message('pars_oi for fit4@sim is not as expected')

fit4dims_oi <- list(y2 = c(4L, 4L), lp__ = integer(0)) 
if (!identical(fit4@sim$dims_oi, fit4dims_oi)) 
  message('dims_oi for fit4@sim is not as expected')

if (!identical(fit4@sim$n_flatnames, as.integer(J * J + 1))) 
  message('n_flatnames for fit4@sim is not as expected')

