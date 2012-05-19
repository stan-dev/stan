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

r <- stanc(stanmodelcode, model_name); 
# cat(r$cppcode, file = 'model.cpp') 
inc <- paste("#include <rstan/nuts_r_ui.hpp>\n", 
             r$cppcode, 
             get_Rcpp_module_def_code(model_name),
             sep = '');

src <- "
  return R_NilValue; 
" 

fx <- cxxfunction(signature(), body = src, include = inc, 
                  # settings = myplugin, verbose = TRUE); 
                  plugin = "rstan", verbose = TRUE)

mod <- Module(model_name, getDynLib(fx)) 

## equivalent to mod$normal1 
modelhmc <- do.call("$", list(mod, model_name))

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

dogsdat_y <- as.integer(dogsdat_y); 

dogsdat <- list(Ndogs = 30L, 
                Ntrials = 25L,
                Y =  structure(as.integer(dogsdat_y), .Dim = c(30, 25))); 

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

model {
  alpha ~ normal(0.0, 316.0);
  beta  ~ normal(0.0, 316.0);
  for(i in 1:Ndogs)
    for (j in 2:Ntrials)
      1 - Y[i, j] ~ bernoulli(exp(alpha * xa[i, j] + beta * xs[i, j]));
}
" 


model_name <- "dogs"; 
dogsr <- stanc(dogsstan, model_name) 

inc <- paste("#include <rstan/nuts_r_ui.hpp>\n", 
             dogsr$cppcode, 
             get_Rcpp_module_def_code(model_name),
             sep = '');

dogsfx <- cxxfunction(signature(), body = src, include = inc, 
                      plugin = "rstan", verbose = TRUE)

dogsmod <- Module(model_name, getDynLib(dogsfx)) 
dogsmodelhmc <- eval(call("$", dogsmod, model_name)) 

dogsb <- new(dogsmodelhmc) 

dogsb$call_nuts(dogsdat, list(sample_file = "dogs.csv")); 

post <- read.csv(file = 'dogs.csv', header = TRUE, skip = 19) 
colMeans(post)
