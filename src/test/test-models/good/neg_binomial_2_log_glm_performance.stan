transformed data {
  int<lower=0> N = 50;
  int<lower=0> M = 100;
  matrix[N,M] x;
  real<lower=0> sigma = 0.5;
  int<lower=0> y[N];
  vector[M] beta_true;
  real alpha_true = 1.5;
  for (j in 1:M)
  {
    beta_true[j] = j/M;
  }
  for (i in 1:N)
  {
    for (j in 1:M)
    {
      x[i,j] = normal_rng(0,1);
    }
    y[i] = neg_binomial_2_log_rng((x * beta_true + alpha_true)[i], sigma);
  }
}
parameters {
  real alpha_inferred;
  vector[M] beta_inferred;
  
}
model {
  beta_inferred ~ normal(0, 2);
  alpha_inferred ~ normal(0, 4);
  
  y ~ neg_binomial_2_log_glm(x, alpha_inferred, beta_inferred, sigma);
}