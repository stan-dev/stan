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
sampling(rr, data = dat, init.t = 0, n.iter = 2012, sample.file = 'norm1.csv')


sampling(rr, data = dat, n.iter = 2012, init.t = 'user', 
         init.v = list(list(mu = 2)), seed = 3, n.thin = 1, 
         sample.file = 'norm1.csv')

post <- read.csv(file = 'norm1.csv', header = TRUE, skip = 19) 
colMeans(post)


### test 2 ################ 
### this should not work since in the list of initial values, 
### mu does not exist. 
### 

sampling(rr, data = dat, n.iter = 2012, init.t = 'user', 
         init.v = list(list(mu2 = 2)), seed = 3, thin = 1, 
         sample.file = 'norm1.csv')

# stan.samples(b, dat, n.chains = 1, n.iter = 2012, 
#              init.t = 'user', init.v = list(mu1 = 2)) 



yasfile <- paste(model_name, ".csv", sep = '')  

sampling(rr, data = dat, n.iter = 2012, init.t = 'random', 
         n.chains = 4,
         seed = 3, n.thin = 1, 
         sample.file = yasfile) 

post <- read.csv(file = yasfile, header = TRUE, skip = 19) 
colMeans(post)



## TODO(mav): 
##   1. s4 class for stan object to do print, plot, etc. 
##   2. include chain object into the c++ code. 
##   3. R doc (Rd) 

