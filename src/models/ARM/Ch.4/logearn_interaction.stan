data {
  int<lower=0> N;
  vector[N] earn;
  vector[N] height;
  vector[N] male;
}
transformed data {
  vector[N] log_earn;        // log transformation
  vector[N] inter;           // interaction
  log_earn <- log(earn);        
  inter    <- height .* male;
}
parameters {
  vector[4] beta;
  real<lower=0> sigma;
}
model {
  log_earn ~ normal(beta[1] + beta[2] * height + beta[3] * male 
                    + beta[4] * inter, sigma);
}
