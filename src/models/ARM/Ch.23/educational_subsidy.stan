data {
  int<lower=0> J;
  int<lower=0> N;
  vector[N] age97;
  vector[N] enroll97;
  vector[N] male;
  vector[N] poor;
  vector[J] program;
  int village[N];
  vector[N] work97;
  int<lower=0,upper=1> y[N];
}
parameters {
  real<lower=0,upper=100> sigma_a;
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
  g_0 ~ norma(0, 100);
  g_1 ~ norma(0, 100);

  a ~ normal(a_hat, sigma_a);
  beta ~ normal(0, 1);
  y ~ bernoulli_logit(y_hat);
}
