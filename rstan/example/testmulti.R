## test fitting multiple stan model in one R session 

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
    real y[J]; 
  } 
  model {
    y ~ uniform(0, 1); 
  }
'

fit1 <- stan(model_code = code1, data = list(N = 3))
fit2 <- stan(model_code = code2, data = list(J = 3))
fit2b <- stan(fit = fit2, data = list(J = 3))
fit1b <- stan(fit = fit1, data = list(J = 3))
