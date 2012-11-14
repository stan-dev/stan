
## test if the dimensions of data and inits are matching between 
## the specified and the declared 

## data 
c1 <- '  
  data {
    real mu[1];
  } 
  parameters {
    real y;
  } 
  model {
    y ~ normal(mu[1], 1);
  } 
'

library(rstan)
fit1 <- stan(model_code = c1, data = list(mu = array(1, dim = 1)), chains = 1)
fit1b <- stan(fit = fit1, data = list(mu = 1), chains = 1)


c2 <- '
  data {
    matrix[4, 5] mu[3];
  }
  parameters {
    matrix[4,5] y[3];
  }
  model {
    for (i in 1:3) for (j in 1:4) for (k in 1:5)
      y[i,j,k] ~ normal(mu[i,j,k], 1);
  }
'

fit2 <- stan(model_code = c2, data = list(mu = array(1:60, dim = c(3, 4, 5))))
fit2b <- stan(fit = fit2, data = list(mu = array(1:60, dim = c(4, 5, 3))))


mu <- array(1:60, dim = c(3, 4, 5))

yinitv <- rnorm(60) 

## inits 
fit2c <- stan(fit = fit2, data = 'mu', init = list(list(y2 = array(yinitv, dim = c(3, 4, 5)))), chains = 1)
fit2d <- stan(fit = fit2, data = 'mu', init = list(list(y = array(yinitv, dim = c(3, 4, 5)))), chains = 1)
fit2e <- stan(fit = fit2, data = 'mu', init = list(list(y = array(yinitv, dim = c(4, 5, 3)))), chains = 1)
