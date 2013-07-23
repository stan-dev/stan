## http://openbugs.info/Examples/Surgical.html
## random effects model 
data {
  int<lower=0> N;
  int r[N];
  int n[N];
}
parameters {
   real mu;
   real<lower=0> sigmasq;
   real b[N];
}
transformed parameters {
  real<lower=0> sigma;
  real<lower=0,upper=1> p[N];
  sigma <- sqrt(sigmasq); 
  for (i in 1:N)
    p[i] <- inv_logit(b[i]);
}
model {
  mu ~ normal(0.0, 1000.0); 
  sigmasq ~ inv_gamma(0.001, 0.001);
  b ~ normal(mu, sigma);
  r ~ binomial_logit(n, b);
}
generated quantities {
  real pop_mean;
  pop_mean <- inv_logit(mu); 
} 
