/**
 * log Jacobian adjustment: log exp'(log(sigma)) = log sigma
 * optimum w/o Jacobian:  sigma = 3
 * optimum w Jacobian: (3 + sqrt(13))/2
 */
parameters {
  real<lower=0> sigma;
}
model {
  sigma ~ normal(3, 1);
}
