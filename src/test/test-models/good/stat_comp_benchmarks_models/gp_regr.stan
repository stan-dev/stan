data {
  int<lower=1> N;
  real x[N];
  vector[N] y;
}

parameters {
  real<lower=0> rho;
  real<lower=0> alpha;
  real<lower=0> sigma;
}

model {
  matrix[N, N] cov =   cov_exp_quad(x, alpha, rho)
                     + diag_matrix(rep_vector(sigma, N));
  matrix[N, N] L_cov = cholesky_decompose(cov);

  rho ~ gamma(25, 4);
  alpha ~ normal(0, 2);
  sigma ~ normal(0, 1);

  y ~ multi_normal_cholesky(rep_vector(0, N), L_cov);
}
