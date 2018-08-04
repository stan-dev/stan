transformed data {
  int N = 2;
  int M = 3;

  int d_y = 1;
  int d_y_a[N] = {1, 0};
  int d_y_da[1] = {1};

  matrix[N,M] d_x_m = [[1, 2, 3],[4, 5, 6]];
  matrix[N,1] d_x_dm1 = [[1],[2]];
  matrix[1,M] d_x_dm2 = [[1, 2, 3]];
  matrix[1,1] d_x_dm3 = [[1]];
  vector[N] d_x_v = [1, 2]';
  vector[1] d_x_dv = [1]';
  row_vector[M] d_x_rv = [1, 2, 3];
  row_vector[1] d_x_drv = [1];

  vector[M] d_beta_v = [1, 2, 3]';
  vector[1] d_beta_dv = [1]';
  row_vector[M] d_beta_rv = [1, 2, 3];
  row_vector[1] d_beta_drv = [1];
  real d_beta_a[M] = {1.0, 2.0, 3.0};
  real d_beta_da[1] = {1.0};
  real d_beta = 1;

  real d_alpha = 3;

  real transformed_data_real;

  transformed_data_real = poisson_log_glm_lpmf(d_y_a| d_x_m, d_beta_v, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y_a| d_x_m, d_beta_rv, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y_a| d_x_m, d_beta_a, d_alpha);

  transformed_data_real = poisson_log_glm_lpmf(d_y_a| d_x_dm1, d_beta_dv, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y_a| d_x_dm1, d_beta_drv, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y_a| d_x_dm1, d_beta_da, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y_a| d_x_dm1, d_beta, d_alpha);

  transformed_data_real = poisson_log_glm_lpmf(d_y| d_x_dm2, d_beta_v, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y| d_x_dm2, d_beta_rv, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y| d_x_dm2, d_beta_a, d_alpha);

  transformed_data_real = poisson_log_glm_lpmf(d_y| d_x_dm3, d_beta_dv, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y| d_x_dm3, d_beta_drv, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y| d_x_dm3, d_beta_da, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y| d_x_dm3, d_beta, d_alpha);

  transformed_data_real = poisson_log_glm_lpmf(d_y_a| d_x_v, d_beta_dv, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y_a| d_x_v, d_beta_drv, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y_a| d_x_v, d_beta_da, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y_a| d_x_v, d_beta, d_alpha);

  transformed_data_real = poisson_log_glm_lpmf(d_y| d_x_dv, d_beta_dv, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y| d_x_dv, d_beta_drv, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y| d_x_dv, d_beta_da, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y| d_x_dv, d_beta, d_alpha);

  transformed_data_real = poisson_log_glm_lpmf(d_y| d_x_rv, d_beta_v, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y| d_x_rv, d_beta_rv, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y| d_x_rv, d_beta_a, d_alpha);

  transformed_data_real = poisson_log_glm_lpmf(d_y| d_x_drv, d_beta_dv, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y| d_x_drv, d_beta_drv, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y| d_x_drv, d_beta_da, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y| d_x_drv, d_beta, d_alpha);

  transformed_data_real = poisson_log_glm_lpmf(d_y_da| d_x_dm2, d_beta_v, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y_da| d_x_dm2, d_beta_rv, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y_da| d_x_dm2, d_beta_a, d_alpha);

  transformed_data_real = poisson_log_glm_lpmf(d_y_da| d_x_dm3, d_beta_dv, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y_da| d_x_dm3, d_beta_drv, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y_da| d_x_dm3, d_beta_da, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y_da| d_x_dm3, d_beta, d_alpha);

  transformed_data_real = poisson_log_glm_lpmf(d_y_da| d_x_dv, d_beta_dv, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y_da| d_x_dv, d_beta_drv, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y_da| d_x_dv, d_beta_da, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y_da| d_x_dv, d_beta, d_alpha);

  transformed_data_real = poisson_log_glm_lpmf(d_y_da| d_x_rv, d_beta_v, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y_da| d_x_rv, d_beta_rv, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y_da| d_x_rv, d_beta_a, d_alpha);

  transformed_data_real = poisson_log_glm_lpmf(d_y_da| d_x_drv, d_beta_dv, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y_da| d_x_drv, d_beta_drv, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y_da| d_x_drv, d_beta_da, d_alpha);
  transformed_data_real = poisson_log_glm_lpmf(d_y_da| d_x_drv, d_beta, d_alpha);
}
parameters {
  matrix[N,M] p_x_m;
  matrix[N,1] p_x_dm1;
  matrix[1,M] p_x_dm2;
  matrix[1,1] p_x_dm3;
  vector[N] p_x_v;
  vector[1] p_x_dv;
  row_vector[M] p_x_rv;
  row_vector[1] p_x_drv;

  vector[M] p_beta_v;
  vector[1] p_beta_dv;
  row_vector[M] p_beta_rv;
  row_vector[1] p_beta_drv;
  real p_beta_a[M];
  real p_beta_da[1];
  real p_beta;

  real p_alpha;

  real y_p;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real = poisson_log_glm_lpmf(d_y_a| p_x_m, p_beta_v, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y_a| p_x_m, p_beta_rv, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y_a| p_x_m, p_beta_a, p_alpha);

  transformed_param_real = poisson_log_glm_lpmf(d_y_a| p_x_dm1, p_beta_dv, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y_a| p_x_dm1, p_beta_drv, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y_a| p_x_dm1, p_beta_da, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y_a| p_x_dm1, p_beta, p_alpha);

  transformed_param_real = poisson_log_glm_lpmf(d_y| p_x_dm2, p_beta_v, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y| p_x_dm2, p_beta_rv, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y| p_x_dm2, p_beta_a, p_alpha);

  transformed_param_real = poisson_log_glm_lpmf(d_y| p_x_dm3, p_beta_dv, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y| p_x_dm3, p_beta_drv, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y| p_x_dm3, p_beta_da, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y| p_x_dm3, p_beta, p_alpha);

  transformed_param_real = poisson_log_glm_lpmf(d_y_a| p_x_v, p_beta_dv, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y_a| p_x_v, p_beta_drv, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y_a| p_x_v, p_beta_da, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y_a| p_x_v, p_beta, p_alpha);

  transformed_param_real = poisson_log_glm_lpmf(d_y| p_x_dv, p_beta_dv, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y| p_x_dv, p_beta_drv, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y| p_x_dv, p_beta_da, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y| p_x_dv, p_beta, p_alpha);

  transformed_param_real = poisson_log_glm_lpmf(d_y| p_x_rv, p_beta_v, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y| p_x_rv, p_beta_rv, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y| p_x_rv, p_beta_a, p_alpha);

  transformed_param_real = poisson_log_glm_lpmf(d_y| p_x_drv, p_beta_dv, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y| p_x_drv, p_beta_drv, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y| p_x_drv, p_beta_da, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y| p_x_drv, p_beta, p_alpha);

  transformed_param_real = poisson_log_glm_lpmf(d_y_da| p_x_dm2, p_beta_v, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y_da| p_x_dm2, p_beta_rv, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y_da| p_x_dm2, p_beta_a, p_alpha);

  transformed_param_real = poisson_log_glm_lpmf(d_y_da| p_x_dm3, p_beta_dv, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y_da| p_x_dm3, p_beta_drv, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y_da| p_x_dm3, p_beta_da, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y_da| p_x_dm3, p_beta, p_alpha);

  transformed_param_real = poisson_log_glm_lpmf(d_y_da| p_x_dv, p_beta_dv, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y_da| p_x_dv, p_beta_drv, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y_da| p_x_dv, p_beta_da, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y_da| p_x_dv, p_beta, p_alpha);

  transformed_param_real = poisson_log_glm_lpmf(d_y_da| p_x_rv, p_beta_v, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y_da| p_x_rv, p_beta_rv, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y_da| p_x_rv, p_beta_a, p_alpha);

  transformed_param_real = poisson_log_glm_lpmf(d_y_da| p_x_drv, p_beta_dv, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y_da| p_x_drv, p_beta_drv, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y_da| p_x_drv, p_beta_da, p_alpha);
  transformed_param_real = poisson_log_glm_lpmf(d_y_da| p_x_drv, p_beta, p_alpha);
}
model {  
  y_p ~ normal(0,1); // in case anyone tries to run it
}
