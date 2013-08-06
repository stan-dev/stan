data {
  int<lower=0> N;
  int<lower=0> n_pair;
  int<lower=0> n_grade;
  int pair[N];
  vector[N] y;
  vector[N] treatment;
  vector[N] pre_test;
  int grade[N];
  int grade_pair[n_pair];
}
parameters {
  vector<lower=0>[n_grade] sigma_y;
  real<lower=0> sigma_b;
  real<lower=0> sigma_c;
  real<lower=0> sigma_a;
  vector[n_grade] eta_b;
  vector[n_grade] eta_c;
  real mu_b;
  real mu_c;
  real mu_a;
  vector[n_pair] eta_a;
}
transformed parameters {
  vector[N] y_hat;
  vector<lower=0>[N] sigma_y_hat;
  vector[n_grade] b;
  vector[n_grade] c;
  vector[n_pair] a;

  a <- mu_a + sigma_a * eta_a;
  b <- mu_b + sigma_b * eta_b;
  c <- mu_c + sigma_c * eta_c;

  for (i in 1:N) {
    y_hat[i] <- a[pair[i]] + b[grade[i]] * treatment[i] 
                 + c[grade[i]] * pre_test[i];
    sigma_y_hat[i] <- sigma_y[grade[i]];
  }
}
model {
  mu_a ~ normal(0, 100);
  mu_b ~ normal(0, 100);
  mu_c ~ normal(0, 100);

  eta_a ~ normal(0, 1);
  eta_b ~ normal(0, 1);
  eta_c ~ normal(0, 1);

  y ~ normal(y_hat, sigma_y_hat);
}
