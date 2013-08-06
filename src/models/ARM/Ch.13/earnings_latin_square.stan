data {
  int<lower=0> N; 
  int<lower=0> n_eth; 
  int<lower=0> n_age; 
  vector[N] y;
  vector[N] x;
  int eth[N];
  int age[N];
} 
transformed data {
  int<lower=0> n_eth_age;
  int eth_age[N];
  
  for (i in 1:N)
    eth_age[i] <- eth[i] * age[i];
  n_eth_age <- n_eth * n_age;
}
parameters {
  matrix[n_eth,2] a;
  matrix[n_age,2] b;
  matrix[n_eth_age,2] c;
  real<lower=0> sigma_y;
  real<lower=0> sigma_a;
  real<lower=0> sigma_b;
  real<lower=0> sigma_c;
  real mu_a;
  real mu_b;
  real mu_c;
}
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] <- a[eth[i],1] + a[eth[i],2] * x[i] + b[age[i],1] 
                + b[age[i],2] * x[i] + c[eth_age[i],1] 
                + c[eth_age[i],2] * x[i];
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
  for (i in 1:n_eth_age)
    c[i] ~ normal (mu_c, sigma_c);

  sigma_y ~ uniform(0, 100);
  y ~ normal(y_hat, sigma_y);
}