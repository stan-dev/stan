

options(error = dump.frames)
library(rstan) 


###########
### dogs example in bugs vol1 

dogsdat_y <- 
  c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0,
    1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0,
    1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0,
    0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0,
    1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0,
    0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1,
    1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1,
    0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1,
    1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1); 

dogsdat <- list(Ndogs = 30, 
                Ntrials = 25,
                Y =  structure(dogsdat_y, .Dim = c(30, 25))); 

dogsstan <- "
data {
  int(0,) Ndogs;
  int(0,) Ntrials;
  int Y[Ndogs, Ntrials];
}

transformed data {
  int xa[Ndogs, Ntrials];
  int xs[Ndogs, Ntrials];
  for (i in 1:Ndogs) {
    for (j in 2 : Ntrials) {
      xs[i, j] <- 0;
      for (k in 1:(j - 1)) xa[i, j] <- xa[i, j] + Y[i, k];
      xs[i, j] <- j - 1 - xa[i, j];
    }
  }
}

parameters {
  real(, -0.00001) alpha;
  real(, -0.00001) beta;
}

transformed parameters {
  real p[2];
  real q[2, 2];
  p[1] <- alpha;
  p[2] <- beta;
  q[1, 1] <- beta;
  q[2, 2] <- alpha;
  q[1, 2] <- 0;
  q[2, 1] <- 1;
} 



model {
  alpha ~ normal(0.0, 316.0);
  beta  ~ normal(0.0, 316.0);
  for(i in 1:Ndogs)
    for (j in 2:Ntrials)
      1 - Y[i, j] ~ bernoulli(exp(alpha * xa[i, j] + beta * xs[i, j]));
}
" 


model_name <- "dogs"; 
dogsrr <- stan.model(model.code = dogsstan, model.name = model_name, 
                     verbose = TRUE) 

ss <- sampling(dogsrr, data = dogsdat, n.chains = 3, seed = 1340338046,
               n.iter = 2012, sample.file = 'dogs.csv')

ss1 <- sampling(dogsrr, data = dogsdat, n.chains = 1, seed = 1340384924,
                n.iter = 2012, sample.file = 'dogs.csv')

  args <- list(init_t = 'random', sample_file = 'dogs.csv', iter = 2012, seed = 1340384924)
  dogsdat <- rstan:::data.preprocess(dogsdat)
  sampler <- new(dogsrr@.modelmod$sampler, dogsdat) 
  t1 <- sampler$call_sampler(args) 
  args$chain_id <- 2;
  t2 <- sampler$call_sampler(args) 
  args$chain_id <- 3;
  t3 <- sampler$call_sampler(args) 
  pnames <- sampler$param_names() 

  pars <- c("alpha", "beta")

args <- attributes(t1)$args 
print(args)

post <- read.csv(file = 'dogs.csv', header = TRUE, skip = 19, comment = "#") 
colMeans(post)

print(ss1)
print(ss)
summary(ss, probs = c(0.25, .5, .75), pars = c('alpha'))
# stop("for debug")

ex <- extract(ss) 
print(ss, pars = c('alpha', 'beta')) 
# print(ss, pars = c('alpha', 'beta1')) # error


sf <- stan(model.code = dogsstan, data = dogsdat, verbose = TRUE, n.chains = 3,
           seed = 1340384924, sample.file = 'dogsb.csv')
traceplot(sf)
plot(sf)
print(sf)

require(coda) 

to.mcmc.list <- function(lst) {
  as.mcmc.list(lapply(lst, FUN = function(x) as.mcmc(do.call(cbind, x))))  
} 
 
tall3 <- to.mcmc.list(sf@sim$samples)
summary(tall3)
effectiveSize(tall3)
gelman.diag(tall3)
