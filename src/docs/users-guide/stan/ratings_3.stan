data {
  int N;
  vector[N] y;
  int J;
  int K;
  int<lower=1, upper=J> movie[N];
  int<lower=1, upper=K> rater[N];
}
parameters {
  vector[J] alpha;
  vector[K] beta;
  real mu;
  real<lower=0> sigma_a;
  real<lower=0> sigma_b;
  real<lower=0> sigma_y;
}
transformed parameters {
  vector[J] a;
  a = mu + sigma_a * alpha;
}
model {
  y ~ normal(mu + sigma_a * alpha[movie] + sigma_b * beta[rater], sigma_y);
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
}
