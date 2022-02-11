data {
  int N;
  array[N] int ids;
  array[N] int y;
  matrix[1000, 11] X;
}
transformed data {
  vector[11] mean_group;
  mean_group = rep_vector(0, 11);
}
parameters {
  array[100] vector[11] beta;
  vector<lower=0>[11] std;
  cholesky_factor_corr[11] L;
}
model {
  vector[N] mu;
  matrix[11, 11] L_std;
  L_std = diag_pre_multiply(std, L);
  for (i in 1 : N) 
    mu[i] = X[i] * beta[ids[i]];
  y ~ bernoulli_logit(mu);
  beta ~ multi_normal_cholesky(mean_group, L_std);
  std ~ normal(0, 1);
  L ~ lkj_corr_cholesky(1);
}

