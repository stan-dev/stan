



Sys.setenv(STAN_HOME = '/home/jq/Desktop/stan') 
LD_LIBRARY_PATH = paste(Sys.getenv("LD_LIBRARY_PATH"), 
                        ":/home/jq/Desktop/stan/bin", 
                        sep = '')

Sys.setenv(LD_LIBRARY_PATH = LD_LIBRARY_PATH)

cat("STAN_HOME=", Sys.getenv("STAN_HOME"), "\n")
cat("LD_LIBRARY_PATH=", Sys.getenv("LD_LIBRARY_PATH"), "\n")

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

require(coda) 

to.mcmc.list <- function(lst) {
  as.mcmc.list(lapply(lst, FUN = function(x) as.mcmc(do.call(cbind, x))))  
} 


model_name <- "dogs"; 
dogsrr <- stan.model(model.code = dogsstan, model.name = model_name, 
                     verbose = TRUE) 

ss <- sampling(dogsrr, data = dogsdat, n.chains = 3, 
               n.iter = 2012, sample.file = 'dogs.csv')



  args <- list(init_t = 'random', sample_file = 'dogs.csv', iter = 2012)
  dogsdat <- rstan:::data.preprocess(dogsdat)
  sampler <- new(dogsrr@.modelmod$sampler, dogsdat, 3)
  sampler$call_sampler(args) 
  args$chain_id <- 2;
  sampler$call_sampler(args) 
  args$chain_id <- 3;
  sampler$call_sampler(args) 
  t1 <- do.call(cbind, sampler$get_chain_samples(1, c("alpha", "beta", "p", "q"))) 
  t2 <- do.call(cbind, sampler$get_chain_samples(2, c("alpha", "beta", "p", "q"))) 
  t3 <- do.call(cbind, sampler$get_chain_samples(3, c("alpha", "beta", "p", "q"))) 
  head(t1)
  pnames <- sampler$param_names() 

  warmup <- sampler$warmup()
  num.s  <- sampler$num_samples() 
  print(sampler$num_chain_samples(1)) 
  k.num.s  <- sampler$num_kept_samples() 
  print(sampler$num_chain_kept_samples(1)) 

  pars <- c("alpha", "beta")

  tall <- sampler$get_samples(pars)
  kepttall <- sampler$get_kept_samples(pars)

probs_oi <- c(0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975)

sq <- sampler$get_quantiles(pars, probs_oi)  
sq2 <- do.call(rbind, sq);

mnsd <- sampler$get_mean_and_sd(pars) 
mnsd1 <- sampler$get_chain_mean_and_sd(1, pars) 
mnsd2 <- sampler$get_chain_mean_and_sd(2, pars) 
mnsd3 <- sampler$get_chain_mean_and_sd(3, pars) 

args <- sampler$get_stan_args(); 
print(args)
args1 <- sampler$get_chain_stan_args(1); 
print(args1)

  tall2 <- lapply(tall, FUN = function(x) do.call(cbind, x)) 
  tall3 <- to.mcmc.list(tall) 

summary(tall3)
effectiveSize(tall3)
gelman.diag(tall3)
  

post <- read.csv(file = 'dogs.csv', header = TRUE, skip = 19, comment = "#") 
colMeans(post)

summary(ss)
summary(ss, probs = c(0.25, .5, .75), pars = c('alpha'))
ex <- extract(ss) 
print(ss, pars = c('alpha', 'beta')) 
print(ss, pars = c('alpha', 'beta1')) # error
