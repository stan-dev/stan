data {
  int<lower=0> N; 
  int<lower=0> n_age; 
  int<lower=0> n_eth; 
  int<lower=1,upper=n_age> age[N];
  int<lower=1,upper=n_eth> eth[N];
  vector[N] x;
  vector[N] y;
} 
parameters {
  matrix[2,n_eth] a;
  matrix[2,n_age] b;
  matrix[n_eth,n_age] c;
  matrix[n_eth,n_age] d;
  real mu_a;
  real mu_b;
  real mu_c;
  real mu_d;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_b;
  real<lower=0,upper=100> sigma_c;
  real<lower=0,upper=100> sigma_d;
  real<lower=0,upper=100> sigma_y;
}
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] <- a[1,eth[i]] + a[2,eth[i]] * x[i] + b[1,age[i]] 
                + b[2,age[i]] * x[i] + c[eth[i],age[i]] 
                + d[eth[i],age[i]] * x[i];
} 
model {
  mu_a ~ normal(0, 1);
  for (i in 1:2)
    a[i] ~ normal(100 * mu_a, sigma_a);

  mu_b ~ normal(0, 1);
  for (i in 1:2)
    b[i] ~ normal (100 * mu_b, sigma_b);

  mu_c ~ normal(0, 1);
  for (i in 1:n_eth)
    c[i] ~ normal (100 * mu_c, sigma_c);

  mu_d ~ normal(0, 1);
  for (i in 1:n_eth)
    d[i] ~ normal (100 * mu_d, sigma_d);

  y ~ normal(y_hat, sigma_y);
}