transformed data {
  int N = 2;
  int M = 3;
  real d_y = 1.0;
  real d_y_a[N] = {4.0, 84.0};
  matrix[N,M] d_x_m = [[1, 2, 3],[4, 5, 6]];
  vector[N] d_x_v = [1, 2]';
  matrix[1, M] d_x_rv = [[1, 2, 3]];
  vector[M] d_beta_v = [1, 2, 3]';
  row_vector[M] d_beta_rv = [1, 2, 3];
  real d_beta_a[M] = {1.0, 2.0, 3.0};
  real d_beta = 2;
  row_vector[1] d_beta_fm = [2];
  real d_beta_fa[1] = {2.0};
  real d_alpha = 3;
  vector[N] d_phi_v = [2, 3]';
  row_vector[N] d_phi_rv = [2, 3];
  real d_phi_a[N] = {2.0, 3.0};
  real d_phi = 2.0;

  real transformed_data_real;

  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_rv, d_beta_v, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_rv, d_beta_rv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_rv, d_beta_a, d_alpha, d_phi);

  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_v, d_beta_fm, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_v, d_beta_fa, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_v, d_beta, d_alpha, d_phi_a);

  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_m, d_beta_v, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_m, d_beta_rv, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_m, d_beta_a, d_alpha, d_phi_a);

  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_v, d_beta_fm, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_v, d_beta_fa, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_v, d_beta, d_alpha, d_phi_v);

  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_m, d_beta_v, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_m, d_beta_rv, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_m, d_beta_a, d_alpha, d_phi_v);

  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_v, d_beta_fm, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_v, d_beta_fa, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_v, d_beta, d_alpha, d_phi_rv);

  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_m, d_beta_v, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_m, d_beta_rv, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_m, d_beta_a, d_alpha, d_phi_rv);
}
parameters {
  real p_y;
  real p_y_a[N];
  matrix[N,M] p_x_m;
  vector[N] p_x_v;
  matrix[1, M] p_x_rv;
  vector[M] p_beta_v;
  row_vector[M] p_beta_rv;
  real p_beta_a[M];
  real p_beta;
  row_vector[1] p_beta_fm;
  real p_beta_fa[1];
  real p_alpha;
  vector[N] p_phi_v;
  row_vector[N] p_phi_rv;
  real p_phi_a[N];
  real p_phi;

  real y_p;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_rv, p_beta_v, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_rv, p_beta_rv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_rv, p_beta_a, p_alpha, p_phi);

  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_v, p_beta_fm, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_v, p_beta_fa, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_v, p_beta, p_alpha, p_phi_a);
  
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_m, p_beta_v, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_m, p_beta_rv, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_m, p_beta_a, p_alpha, p_phi_a);

  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_v, p_beta_fm, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_v, p_beta_fa, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_v, p_beta, p_alpha, p_phi_v);
  
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_m, p_beta_v, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_m, p_beta_rv, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_m, p_beta_a, p_alpha, p_phi_v);

  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_v, p_beta_fm, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_v, p_beta_fa, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_v, p_beta, p_alpha, p_phi_v);
  
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_m, p_beta_v, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_m, p_beta_rv, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_m, p_beta_a, p_alpha, p_phi_v);
}
model {  
  y_p ~ normal(0,1); // in case anyone tries to run it
}
