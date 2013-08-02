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
  vector<lower=0>[n_grade] sigma_a;
  real<lower=0> sigma_beta;
  vector[n_grade] beta;
  vector[n_grade] mu_a;
  real mu_beta;
  vector[n_pair] a;
}
transformed parameters {
  vector[N] y_hat;
  vector<lower=0>[N] sigma_y_hat;
  vector[n_pair] sigma_a_hat;
  vector[n_pair] mu_a_hat;

  for (i in 1:N) {
    y_hat[i] <- a[pair[i]] + beta[grade[i]] * treatment[i];
    sigma_y_hat[i] <- sigma_y[grade[i]];
  }

  for (i in 1:n_pair) {
    sigma_a_hat[i] <- sigma_a[grade_pair[i]];
    mu_a_hat[i] <- mu_a[grade_pair[i]];
  }
}
model {
  mu_a ~ normal(0, .0001);
  sigma_a ~ uniform(0, 100);
  mu_beta ~ normal(0, .0001);
  sigma_beta ~ uniform(0, 100);
  sigma_y ~ uniform(0, 100);

  a ~ normal(mu_a_hat, sigma_a_hat);
  beta ~ normal(mu_beta, sigma_beta);
  y ~ normal(y_hat, sigma_y_hat);
}
