data {
  int<lower=0> N;              // num individuals
  int<lower=1> K;              // num ind predictors
  int<lower=1> J;              // num groups
  int<lower=1> L;              // num group predictors
  int<lower=1,upper=J> jj[N];  // group for individual
  matrix[N,K] x;               // individual predictors
  matrix[J,L] u;               // group predictors
  vector[N] y;                 // outcomes
}
parameters {
  corr_matrix[K] Omega;     // prior covariance
  vector<lower=0>[K] tau;   // prior scale
  matrix[L,K] gamma;        // group coeffs
  vector[K] beta[J];        // indiv coeffs by group
  real<lower=0> sigma;      // prediction error scale
}
model {
  matrix[K,K] Sigma_beta;
  Sigma_beta <- diag_matrix(tau) * Omega * diag_matrix(tau);

  tau ~ cauchy(0,2.5);
  Omega ~ lkj_corr(2);
  for (l in 1:L)
    gamma[l] ~ normal(0,5);

  for (j in 1:J)
    beta[j] ~ multi_normal((u[j] * gamma)', Sigma_beta);

  for (n in 1:N)
    y[n] ~ normal(x[n] * beta[jj[n]], sigma);
}
