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

modelhmc <- rr@.modelData[["nuts"]] 


y <- rnorm(20) 
mean(y) 
sd(y)
dat <- list(N = 20L, y = y); 
b <- new(modelhmc) # , dat, list(a = 3)) 
## TODO(mav): 
##   1. add data check and as.integer for dat
##   2. check args list, for example postiveness of 
##      some arguments.  
##   3. wrap b$call_nuts to a R function: creating the hmc_args list 
##      from given R function's parameters, which should be more friendly. 
##   4. s4 class for stan object to do print, plot, etc. 
##   5. include chain object into the c++ code. 
##   6. R doc (Rd) 

b$call_nuts(dat, 
            list(iter = 2012, 
                 seed = 3, 
                 thin = 1, 
                 init = 'user', 
                 init_lst = list(mu = 2))) 

post <- read.csv(file = 'samples.csv', header = TRUE, skip = 19) 
colMeans(post)


### test 2 ################ 
### this should not work since in the list of initial values, 
### mu does not exist. 
### 
b$call_nuts(dat, 
            list(iter = 2012, 
                 seed = 3, 
                 thin = 1, 
                 init = 'user', 
                 init_lst = list(mu1 = 2))) 
################### 

b$call_nuts(dat, 
            list(iter = 2012, 
                 seed = 3, 
                 thin = 1, 
                 init = '0'));  
yasfile <- paste(model_name, ".csv", sep = '')  

b$call_nuts(dat, 
            list(iter = 2012, 
                 sample_file = yasfile,
                 seed = 3, 
                 thin = 1, 
                 init = 'random'));  

post <- read.csv(file = yasfile, header = TRUE, skip = 19) 
colMeans(post)


