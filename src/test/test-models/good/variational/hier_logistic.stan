data {
  int N;
  int ids[N];
  int y[N];
  matrix[1000,11] X;
}
parameters {
  matrix[11,100] beta_std;
  vector<lower=0>[11] std;
  cholesky_factor_corr[11] L;
}
transformed parameters {
  matrix[100,11] beta;
  beta <- (diag_pre_multiply(std,L) * beta_std)';
}
model {
  vector[N] mu;
  for(i in 1:N)
    mu[i] <- X[i] * beta[ids[i]]';
  y ~ bernoulli_logit(mu);
  to_vector(beta_std) ~ normal(0,1);
  std ~ normal(0,1);
  L ~ lkj_corr_cholesky(0.5);
}
