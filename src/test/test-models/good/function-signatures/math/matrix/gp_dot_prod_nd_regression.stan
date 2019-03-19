functions {
  vector gp_pred_rng(vector[] x_pred,
                     vector y1, vector[] x,
                     real sig0,
                     real sigma) {
    int N = rows(y1);
    int N_pred = size(x_pred);
    vector[N_pred] f2;
    {
      matrix[N, N] K = gp_dot_prod_cov(x, 1.0) +
        diag_matrix(rep_vector(square(sigma), N));
      matrix[N, N] L_K = cholesky_decompose(K);
      vector[N] L_K_div_y1 = mdivide_left_tri_low(L_K, y1);
      vector[N] K_div_y1 = mdivide_right_tri_low(L_K_div_y1', L_K)';
      matrix[N, N] k_x_x_pred = gp_dot_prod_cov(x, x_pred, 1.0);
      f2 = (k_x_x_pred' * K_div_y1);
    }
    return f2;
  }
}
data {
  int<lower=1> N;
  int<lower=1> D;
  vector[D] x[N];
  vector[N] y;

  int<lower=1> N_pred;
  vector[D] x_pred[N_pred];
}
transformed data {
  vector[N] mu;
  mu = rep_vector(0, N);
}
parameters {
  real<lower=0> magnitude;
  real<lower=0> length_scale;
  real<lower=0> sig0;
  
  real<lower=0> sigma;
}
transformed parameters {
  matrix[N, N] L_K;
  {
    matrix[N, N] K = gp_dot_prod_cov(x, sig0);
    for(i in 1:N) K[i, i] = K[i, i] + square(sigma);
    L_K = cholesky_decompose(K);
  }
}
model {
  sig0 ~ normal(0, 1);
  
  sigma ~ normal(0, 1);
  
  y ~ multi_normal_cholesky(mu, L_K);
}
generated quantities {
  vector[N_pred] f_pred = gp_pred_rng(x_pred, y, x, sig0, sigma);
  vector[N_pred] y_pred;
  for (n in 1:N_pred) y_pred[n] = normal_rng(f_pred[n], sigma);
}
