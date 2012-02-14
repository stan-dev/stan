
## http://openbugs.info/Examples/Surgical.html

## random effects model 
data {
  int(0,) N;
  int r[N];
  int n[N];
}
parameters {
   real b[N];
   real mu;
   real(0,) sigmasq;
}
transformed parameters {
  real(0,) sigma; 
  sigma <- sqrt(sigmasq); 
}
model {
  b ~ normal(mu, sigma);
  for (i in 1:N) 
    r[i] ~ binomial(n[i], inv_logit(b[i]));
  mu ~ normal(0.0, 1000); 
  sigmasq ~ inv_gamma(0.001, 0.001);
}

generated quantities {
  real pop_mean;
  pop_mean <- inv_logit(mu); 
} 
