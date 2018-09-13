transformed data {
  int N = 2;
  int M = 3;

  vector[N] d_y_v = [1.0, 0.0]';

  matrix[N,M] d_x_m = [[1, 2, 3],[4, 5, 6]];

  vector[M] d_beta_v = [1, 2, 3]';

  real d_alpha = 3;
  vector[N] d_alpha_v = [0.5, 0.6]';

  real d_phi = 2;

  real transformed_data_real;

  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_m, d_alpha, d_beta_v, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_m, d_alpha_v, d_beta_v, d_phi);
}
parameters {
  vector[N] p_y_v;

  matrix[N,M] p_x_m;

  vector[M] p_beta_v;

  real p_alpha;
  vector[N] p_alpha_v;

  real<lower=0> p_phi;

  real y_p;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_m, p_alpha, p_beta_v, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_m, p_alpha_v, p_beta_v, p_phi);
}
model {  
  y_p ~ normal(0,1); // in case anyone tries to run it
}
