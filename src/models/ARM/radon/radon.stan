data {
  int<lower=0> N; 
  int<lower=0> J;
  vector[N] y;       
  int county[N];
  vector[N] x;       
} 
parameters {
  vector[J] a;
  real b;
  real mu_a;
  real sigma_y;
  real sigma_a;
} 
transformed parameters {
  real tau_y;
  real tau_a;
  tau_y <- pow(sigma_y,-2);
  tau_a <- pow(sigma_a,-2);
}
model {
  b ~ normal(0, 0.0001);
  mu_a ~ normal(0,0.0001);
  sigma_y ~ uniform(0,100);
  sigma_a ~ uniform(0,100);
  a ~ normal(mu_a,tau_a);
  for (n in 1:N)
    y[n] ~ normal (a[county[n]] + b * x[n],tau_y);
}
