transformed data {
  int<lower=0> N = 50;
  int<lower=0> M = 100;
  matrix[N,M] x;
  int<lower=0,upper=1> y[N];
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