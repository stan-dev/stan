functions {
  vector gp_pred_rng(real[] x_pred,
                     vector y1, real[] x,
                     real magnitude, real length_scale,
                     real sigma) {
    int N = rows(y1);
    int N_pred = size(x_pred);
    vector[N_pred] f2;
    {
      matrix[N, N] K = gp_matern32_cov(x, magnitude, length_scale) +
        diag_matrix(rep_vector(square(sigma), N));
      matrix[N, N] L_K = cholesky_decompose(K);
      vector[N] L_K_div_y1 = mdivide_left_tri_low(L_K, y1);
      vector[N] K_div_y1 = mdivide_right_tri_low(L_K_div_y1', L_K)';
      matrix[N, N_pred] k_x_x_pred = gp_matern32_cov(x, x_pred, magnitude, length_scale);
      f2 = (k_x_x_pred' * K_div_y1);
    }
    return f2;
  }
}
data {
  int<lower=1> D;
  int<lower=1> N;
  real x[N];
  vector[N] y;

  int<lower=1> N_pred;
  real x_pred[N_pred];
}
transformed data {
  vector[N] mu;
  mu = rep_vector(0, N);
}
parameters {
  real<lower=0> magnitude;
  real<lower=0> length_scale;
  real<lower=0> sig;
  
  real<lower=0> sigma;
}
transformed parameters {
  matrix[N, N] L_K;
  {
    matrix[N, N] K = gp_matern32_cov(x, magnitude, length_scale);
    for(i in 1:N) K[i, i] = K[i, i] + square(sigma);
    L_K = cholesky_decompose(K);
  }
}
model {
  magnitude ~ normal(0, 3);
  length_scale ~ normal(1, 3);
  sig ~ normal(0, 1);
  
  sigma ~ normal(0, 1);
  
  y ~ multi_normal_cholesky(mu, L_K);
}
generated quantities {
  vector[N_pred] f_pred = gp_pred_rng(x_pred, y, x, magnitude, length_scale, sigma);
  vector[N_pred] y_pred;
  for (n in 1:N_pred) y_pred[n] = normal_rng(f_pred[n], sigma);
}
