# test warning for integer division

library(rstan)


scode <- '
transformed data {
  int x;
  x <- 5 / 2;
}
parameters {
  real y;
} 
model {
  y ~ normal(x, 1);
}
'


a <- stan(model_code = scode, iter = 10) 

