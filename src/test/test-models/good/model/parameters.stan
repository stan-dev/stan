// used to test parameter serialization/deserialization
parameters {
  real theta;
  tuple(real, real) t;
  array[3] real<lower=0> sigma;
  vector[3] mu;
  array[3] tuple(array[3] real, tuple (simplex[2], real)) arr_t;
  array[2, 3] simplex[4] alpha;
  complex_matrix[3, 4] cm;
  cholesky_factor_cov[3] L_Omega;
  cholesky_factor_corr[3] L_Corr;
  array[2, 3, 2] corr_matrix[3] Omega;
  array[1, 2, 3] complex_vector[4] cv;
  real<lower=0, upper=1> p;
}
transformed parameters {
  vector[3] mu2;
  mu2 = mu + 1;
  tuple(array[3] real, vector[3]) t2 = (sigma, mu2);
}
generated quantities {
  array[3] real y;
  y = normal_rng(mu2, sigma);
  tuple(array[2, 3] simplex[4], real<lower=0, upper=1>) t3 = (alpha, p);
}
