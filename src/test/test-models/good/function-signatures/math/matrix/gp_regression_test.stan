functions {}
data {
  int<lower=1> D;
  int<lower=1> N;
  vector[D] x1[N];
  vector[N] y;
  real x[N];
  
  int<lower=1> N_pred;
  vector[D] x1_pred[N_pred];
}
transformed data {
  vector[N] mu;
  mu = rep_vector(0, N);
}
parameters {
  real<lower=0> magnitude;
  real<lower=0> length_scale;
  real<lower=0> length_scale_ard[D];
  real<lower=0> sig;
  
  real<lower=0> sigma;
}
transformed parameters {
  matrix[N, N] L_K;
  {
    // dot product covariance matrix
    matrix[N, N] K =
      gp_dot_prod_cov(x, sig);  // dot product kernel 1-D
    matrix[N, N] K2 =
      gp_dot_prod_cov(x1, sig);  // dot product N-D
    matrix[N, N] K3 =
      gp_dot_prod_cov(x, x, sig);   // dot product cross covariance 1-D
    matrix[N, N] K4 =
      gp_dot_prod_cov(x1, x1, sig);   // dot product cross covariance N-D
    // RBF   
    matrix[N, N] K5 =
      gp_exp_quad_cov(x, magnitude, length_scale);  // 1-D
    matrix[N, N] K6 =
      gp_exp_quad_cov(x1, magnitude, length_scale);  // N-D
    matrix[N, N] K7 =
      gp_exp_quad_cov(x, x, magnitude, length_scale);  // 1-D cross covariance
    matrix[N, N] K8 =
      gp_exp_quad_cov(x1, x1, magnitude, length_scale);  // N-D cross covariance
    matrix[N, N] K17 =
      gp_exp_quad_cov(x1, magnitude, length_scale_ard);  // N-D ard
    matrix[N, N] K18 =
      gp_exp_quad_cov(x1, x1, magnitude, length_scale_ard);  // N-D ard cross covariance
    
    // matern32
    matrix[N, N] K9 =
      gp_matern32_cov(x, magnitude, length_scale);  // 1-D
    matrix[N, N] K10 =
      gp_matern32_cov(x1, magnitude, length_scale);  // N-D
    matrix[N, N] K11 =
      gp_matern32_cov(x, x, magnitude, length_scale);  // 1-D cross covariance
    matrix[N, N] K12 =
      gp_matern32_cov(x1, x1, magnitude, length_scale);  // N-D cross covariance
    matrix[N, N] K19 =
      gp_matern32_cov(x1, magnitude, length_scale_ard);  // N-D ard
    matrix[N, N] K20 =
      gp_matern32_cov(x1, x1, magnitude, length_scale_ard);  // N-D ard cross covariance

    // matern52
    matrix[N, N] K13 =
      gp_matern52_cov(x, magnitude, length_scale);  // 1-D
    matrix[N, N] K14 =
      gp_matern52_cov(x1, magnitude, length_scale);  // N-D
    matrix[N, N] K15 =
      gp_matern52_cov(x, x, magnitude, length_scale);  // 1-D cross covariance
    matrix[N, N] K16 =
      gp_matern52_cov(x1, x1, magnitude, length_scale);  // N-D cross covariance
    matrix[N, N] K21 =
      gp_matern52_cov(x1, magnitude, length_scale_ard);  // N-D ard
    matrix[N, N] K22 =
      gp_matern52_cov(x1, x1, magnitude, length_scale_ard);  // N-D ard cross covariance
    // exponential
    matrix[N, N] K23 =
      gp_exponential_cov(x, magnitude, length_scale);  // 1-D
    matrix[N, N] K24 =
      gp_exponential_cov(x1, magnitude, length_scale);  // N-D
    matrix[N, N] K55 =
      gp_exponential_cov(x, x, magnitude, length_scale);  // 1-D cross covariance
    matrix[N, N] K26 =
      gp_exponential_cov(x1, x1, magnitude, length_scale);  // N-D cross covariance
    matrix[N, N] K27 =
      gp_exponential_cov(x1, magnitude, length_scale_ard);  // N-D ard
    matrix[N, N] K28 =
      gp_exponential_cov(x1, x1, magnitude, length_scale_ard);  // N-D ard cross covariance
    // periodic
    matrix[N, N] K29 =
      gp_periodic_cov(x, magnitude, length_scale, 1234);  // 1-D
    matrix[N, N] K30 =
      gp_periodic_cov(x, x, magnitude, length_scale, 1234);  // 1-D cross covariance
    matrix[N, N] K31 =
      gp_periodic_cov(x1, magnitude, length_scale, 121);  // N-D
    matrix[N, N] K32 =
      gp_periodic_cov(x1, x1, magnitude, length_scale, 121);  // N-D cross covariance
  }
}
model {}
generated quantities {}
