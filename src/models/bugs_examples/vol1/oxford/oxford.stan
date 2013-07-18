# http://www.mrc-bsu.cam.ac.uk/bugs/winbugs/Vol1.pdf
# Page 34: Oxford: smooth fit to log-odds ratios

data {
  int<lower=0> K; 
  int<lower=0> n0[K];
  int<lower=0> n1[K]; 
  int<lower=0> r0[K]; 
  int<lower=0> r1[K]; 
  vector[K] year;
} 
transformed data {
  vector[K] yearsq;
  yearsq <- year .* year;
} 
parameters {
  vector[K] mu;
  real alpha;
  real beta1; 
  real beta2;
  real<lower=0> sigma_sq;
  vector[K] b; 
}
transformed parameters {
  real<lower=0> sigma;
  sigma <- sqrt(sigma_sq);
}
model {
  r0 ~ binomial_logit(n0, mu); 
  r1 ~ binomial_logit(n1, alpha + mu + beta1 * year + beta2 * (yearsq - 22) + b * sigma); 
  b  ~ normal(0, 1);
  mu ~ normal(0, 1000); 

  alpha  ~ normal(0.0, 1000); 
  beta1  ~ normal(0.0, 1000); 
  beta2  ~ normal(0.0, 1000); 
  sigma_sq ~ inv_gamma(0.001, 0.001);
}
