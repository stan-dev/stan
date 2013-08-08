data {
  int<lower=0> N; 
  int<lower=0> n_eth; 
  int<lower=0> n_age; 
  vector[N] y;
  vector[N] x;
  int eth[N];
  int age[N];
} 
parameters {
  matrix[n_eth,2] a;
  matrix[n_age,2] b;
  matrix[n_eth,n_age] c;
  matrix[n_eth,n_age] d;
  real<lower=0> sigma_y;
  real<lower=0> sigma_a;
  real<lower=0> sigma_b;
  real<lower=0> sigma_c;
  real<lower=0> sigma_d;
  real mu_a;
  real mu_b;
  real mu_c;
  real mu_d;
}
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] <- a[eth[i],1] + a[eth[i],2] * x[i] + b[age[i],1] 
                + b[age[i],2] * x[i] + c[eth[i],age[i]] 
                + d[eth[i],age[i]] * x[i];
} 
model {
  mu_a ~ normal(0, 100);
  sigma_a ~ uniform(0, 100);
  for (i in 1:n_eth)
    a[i] ~ normal(mu_a, sigma_a);

  mu_b ~ normal(0, 100);
  sigma_b ~ uniform(0, 100);
  for (i in 1:n_age)
    b[i] ~ normal (mu_b, sigma_b);

  mu_c ~ normal(0, 100);
  sigma_c ~ uniform(0, 100);
  for (i in 1:n_eth)
    c[i] ~ normal (mu_c, sigma_c);

  mu_d ~ normal(0, 100);
  sigma_d ~ uniform(0, 100);
  for (i in 1:n_eth)
    d[i] ~ normal (mu_d, sigma_d);

  sigma_y ~ uniform(0, 100);
  y ~ normal(y_hat, sigma_y);
}