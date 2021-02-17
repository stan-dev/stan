transformed data {
  real rho = 5.5;
  real alpha = 3;
  real sigma = 2;
}
parameters {

}
model {

}
generated quantities {
  int<lower=1> N = 11;
  array[11] real x = {-10, -8, -6, -4, -2, 0.0, 2, 4, 6, 8, 10};
  vector[11] y;
  array[11] int k;
  {
    matrix[N, N] cov = gp_exp_quad_cov(x, alpha, rho)
                       + diag_matrix(rep_vector(1e-10, N));
    matrix[N, N] L_cov = cholesky_decompose(cov);
    vector[N] f = multi_normal_cholesky_rng(rep_vector(0, N), L_cov);
    for (n in 1 : N) {
      y[n] = normal_rng(f[n], sigma);
      k[n] = poisson_rng(exp(f[n]));
    }
  }
}

