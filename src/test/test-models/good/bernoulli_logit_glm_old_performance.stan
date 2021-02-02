Info: Found int division at 'src/test/test-models/good/bernoulli_logit_glm_old_performance.stan', line 10, column 19 to column 20:
  j / M
Values will be rounded towards zero. If rounding is not desired you can write
the division as
  j * 1.0 / M
transformed data {
  int<lower=0> N = 50;
  int<lower=0> M = 100;
  matrix[N, M] x;
  array[N] int<lower=0, upper=1> y;
  vector[M] beta_true;
  real alpha_true = 1.5;
  for (j in 1 : M) {
    beta_true[j] = j / M;
  }
  for (i in 1 : N) {
    for (j in 1 : M) {
      x[i, j] = normal_rng(0, 1);
    }
    y[i] = bernoulli_logit_rng((x * beta_true + alpha_true)[i]);
  }
}
parameters {
  real alpha_inferred;
  vector[M] beta_inferred;
}
model {
  beta_inferred ~ normal(0, 2);
  alpha_inferred ~ normal(0, 4);
  y ~ bernoulli_logit(x * beta_inferred + alpha_inferred);
}

If rounding is intended please use the integer division operator %/%.