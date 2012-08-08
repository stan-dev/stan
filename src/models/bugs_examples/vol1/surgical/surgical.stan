## http://openbugs.info/Examples/Surgical.html
## random effects model 
data {
  int[0,] N;
  int r[N];
  int n[N];
}
parameters {
   real mu;
   real[0,] sigmasq;
   real b[N];
}
transformed parameters {
  real[0,] sigma;
  real[0,1] p[N];
  sigma <- sqrt(sigmasq); 
  for (i in 1:N)
    p[i] <- inv_logit(b[i]);
}
model {
  mu ~ normal(0.0, 1000.0); 
  sigmasq ~ inv_gamma(0.001, 0.001);
  b ~ normal(mu, sigma);
  for (i in 1:N) 
    r[i] ~ binomial(n[i], inv_logit(b[i]));
}
generated quantities {
  real pop_mean;
  pop_mean <- inv_logit(mu); 
} 
