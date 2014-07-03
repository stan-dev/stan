data {
  int<lower=0> N; 
  int<lower=1,upper=85> county[N];
  vector[N] u;
  vector[N] x;
  vector[N] y;
} 
transformed data {
  vector[N] inter;

  inter <- u .* x;
}
parameters {
  vector[2] beta;
  vector[85] eta1;
  vector[85] eta2;
  real mu_a1;
  real mu_a2;
  real<lower=0,upper=100> sigma_a1;
  real<lower=0,upper=100> sigma_a2;
  real<lower=0,upper=100> sigma_y;
} 
transformed parameters {
  vector[85] a1;
  vector[85] a2;
  vector[N] y_hat;

  a1 <- mu_a1 + sigma_a1 * eta1;
  a2 <- 0.1 * mu_a2 + sigma_a2 * eta2;

  for (i in 1:N)
    y_hat[i] <- a1[county[i]] + x[i] * a2[county[i]] + beta[1] * u[i] 
                + beta[2] * inter[i];
}
model {
  beta ~ normal(0, 100);

  mu_a1 ~ normal(0, 1);
  mu_a2 ~ normal(0, 1);
  eta1 ~ normal(0, 1);
  eta2 ~ normal(0, 1);

  y ~ normal(y_hat, sigma_y);
}
