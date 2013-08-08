data {
  int<lower=0> N;
  int<lower=0> n_grade;
  int<lower=0> n_grade_pair;
  int<lower=0> n_pair;
  int<lower=1,upper=n_grade> grade[N];
  int<lower=1,upper=n_grade_pair> grade_pair[n_pair];
  int<lower=1,upper=n_pair> pair[N];
  vector[N] treatment;
  vector[N] y;
}
parameters {
  vector[n_grade_pair] eta_a;
  vector[n_grade] eta_b;
  vector<lower=0,upper=100>[n_grade_pair] sigma_a;
  real<lower=0,upper=100> sigma_b;
  vector<lower=0,upper=100>[n_grade] sigma_y;
  vector[n_grade_pair] mu_a;
  real mu_b;
}
transformed parameters {
  vector[n_grade_pair] a;
  vector[n_grade] b;
  vector<lower=0>[N] sigma_y_hat;
  vector[N] y_hat;

  a <- 100 * mu_a + sigma_a .* eta_a;
  b <- 100 * mu_b + sigma_b * eta_b;

  for (i in 1:N) {
    y_hat[i] <- a[grade[pair[i]]] + b[grade[i]] * treatment[i];
    sigma_y_hat[i] <- sigma_y[grade[i]];
  }
}
model {
  mu_a ~ normal(0, 1);
  mu_b ~ normal(0, 1);

  eta_a ~ normal(0, 1);
  eta_b ~ normal(0, 1);

  y ~ normal(y_hat, sigma_y_hat);
}
