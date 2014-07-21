data {
  int<lower=0> N;
  int<lower=0> J;
  vector[N] y;
  int<lower=0,upper=1> x[N];
  int county[N];
}
parameters {
  real a[J];
  real b;              
  real<lower=0> sigma_y;
}
model {
  for (i in 1:N)
    y[i] ~ normal(a[county[i]] + b * x[i], sigma_y);
}
