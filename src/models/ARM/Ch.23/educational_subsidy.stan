data {
  int<lower=0> N;
  int<lower=0> J;
  vector[N] y;
  vector[N] enroll97;
  vector[N] work97;
  vector[N] poor;
  vector[N] male;
  vector[N] age97;
  int village[N];
  vector[J] program;
}
parameters {
  real<lower=0> sigma_a;
  real<lower=0> sigma_beta;
  vector[5] beta;
  real mu_beta;
  vector[J] a;
}
transformed parameters {
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] <- a[village[i]] + beta[1] * enroll97[i] + beta[2] * work97[i]
                  + beta[3] * poor[i] + beta[4] * male[i] + beta[5] * age97[i];
  for (j in 1:J)
    a_hat[j] <- g_0 + g_1 * program[j];
}
model {
  sigma_a ~ uniform(0, 100);
  mu_beta ~ normal(0, .0001);
  sigma_beta ~ uniform(0, 100);
  g_0 ~ norma(0, .0001);
  g_1 ~ norma(0, .0001);

  a ~ normal(a_hat, sigma_a);
  beta ~ normal(mu_beta, sigma_beta);
  y ~ bernoulli_logit(y_hat);
}
