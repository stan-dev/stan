data {
  int<lower=0> N;
  int<lower=0> J;
  vector[N] y;
  int<lower=0,upper=1> x[N];
  int county[N];
  vector[J] u;
}
transformed data {
  real<lower=0> sigma_y;
  real<lower=0> sigma_a;
  sigma_y <- 0.7;
  sigma_a <- 0.4;
}
parameters {
  real a[J];
  real b;
  real g_0;
  real g_1;
}
model {
  for (j in 1:J)
    a[j] ~ normal(g_0 + g_1 * u[j], sigma_a);
  for (n in 1:N)
    y[n] ~ normal(a[county[n]] + b * x[n], sigma_y);
}  
