library(rstan)
# example(stanc)


stanmodelcode <- '
data {
  int(0,) N;
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

rr <- stan.model(model.code = stanmodelcode, model.name = model_name, 
                 verbose = TRUE) 

y <- rnorm(20) 
mean(y) 
sd(y)
dat <- list(N = 20, y = y); 
samples(rr, data = dat, n.iter = 2012, sample_file = 'norm1.csv')


samples(rr, data = dat, n.iter = 2012, init.t = 'user', 
        init.v = list(mu = 2), seed = 3, thin = 1, 
        sample_file = 'norm1.csv')

post <- read.csv(file = 'norm1.csv', header = TRUE, skip = 19) 
colMeans(post)


### test 2 ################ 
### this should not work since in the list of initial values, 
### mu does not exist. 
### 

samples(rr, data = dat, n.iter = 2012, init.t = 'user', 
        init.v = list(mu2 = 2), seed = 3, thin = 1, 
        sample_file = 'norm1.csv')

# stan.samples(b, dat, n.chains = 1, n.iter = 2012, 
#              init.t = 'user', init.v = list(mu1 = 2)) 


yasfile <- paste(model_name, ".csv", sep = '')  

samples(rr, data = dat, n.iter = 2012, init.t = 'random', 
        seed = 3, thin = 1, 
        sample_file = yasfile) 

post <- read.csv(file = yasfile, header = TRUE, skip = 19) 
colMeans(post)



## TODO(mav): 
##   1. add data check and as.integer for dat
##   2. check args list, for example postiveness of 
##      some arguments.  
##   3. wrap b$call_nuts to a R function: creating the hmc_args list 
##      from given R function's parameters, which should be more friendly. 
##   4. s4 class for stan object to do print, plot, etc. 
##   5. include chain object into the c++ code. 
##   6. R doc (Rd) 

