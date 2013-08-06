data {
  int<lower=0> N;
  int<lower=0> n_pair;
  int<lower=0> n_grade;
  int pair[N];
  vector[N] y;
  vector[N] treatment;
  int grade[N];
  int grade_pair[n_pair];
}
parameters {
  vector<lower=0>[n_grade] sigma_y;
  real<lower=0> sigma_a;
  real<lower=0> sigma_b;
  vector[n_grade] eta_b;
  real mu_a;
  real mu_b;
  vector[n_pair] eta_a;
}
transformed parameters {
  vector[N] y_hat;
  vector<lower=0>[N] sigma_y_hat;
  vector[n_pair] a;
  vector[n_grade] b;

  a <- mu_a + sigma_a * eta_a;
  b <- mu_b + sigma_b * eta_b;

  for (i in 1:N) {
    y_hat[i] <- a[pair[i]] + b[grade[i]] * treatment[i];
    sigma_y_hat[i] <- sigma_y[grade[i]];
  }
}
model {
  mu_a ~ normal(0, 100);
  mu_b ~ normal(0, 100);
  eta_a ~ normal(0, 1);
  eta_b ~ normal(0, 1);

  y ~ normal(y_hat, sigma_y_hat);
}
