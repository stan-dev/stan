data {
  int<lower=0> N;
  int<lower=0> n_pair;
  int<lower=0> n_grade;
  int<lower=0> n_grade_pair;
  int<lower=1,upper=n_grade> grade[N];
  int<lower=1,upper=n_grade_pair> grade_pair[n_pair];
  int<lower=1,upper=n_pair> pair[N];
  vector<lower=0,upper=1>[N] treatment;
  vector[N] y;
}
parameters {
  vector[n_pair] a;
  vector[n_grade] b;
  vector[n_grade_pair] mu_a;
  vector<lower=0,upper=100>[n_grade_pair] sigma_a;
  vector<lower=0,upper=100>[n_grade] sigma_y;
}
transformed parameters {
  vector[n_pair] mu_a_hat;
  vector<lower=0,upper=100>[n_pair] sigma_a_hat;
  vector<lower=0,upper=100>[N] sigma_y_hat;
  vector[N] y_hat;

  for (i in 1:N) {
    y_hat[i] <- a[pair[i]] + b[grade[i]] * treatment[i];
    sigma_y_hat[i] <- sigma_y[grade[i]];
  }

  for (i in 1:n_pair) {
    sigma_a_hat[i] <- sigma_a[grade_pair[i]];
    mu_a_hat[i] <- 100 * mu_a[grade_pair[i]];
  }
}
model {
  mu_a ~ normal(0, 1);

  a ~ normal(mu_a_hat, sigma_a_hat);
  b ~ normal(0, 100);
  y ~ normal(y_hat, sigma_y_hat);
}
