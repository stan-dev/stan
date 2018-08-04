transformed data {
  int N = 2;
  int M = 3;

  real d_y = 1;
  real d_y_a[N] = {1.0, 0.0};
  real d_y_da[1] = {1.0};
  vector[N] d_y_v = [1.0, 0.0]';
  vector[1] d_y_dv = [1.0]';
  row_vector[N] d_y_rv = [1.0, 0.0];
  row_vector[1] d_y_drv = [1.0];

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

  vector[N] d_phi_v = [1, 2]';
  vector[1] d_phi_dv = [1]';
  row_vector[N] d_phi_rv = [1, 2];
  row_vector[1] d_phi_drv = [1];
  real d_phi_a[N] = {1.0, 2.0};
  real d_phi_da[1] = {1.0};
  real d_phi = 1;

  real transformed_data_real;

  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_m, d_beta_v, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_m, d_beta_rv, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_m, d_beta_a, d_alpha, d_phi_v);

  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_m, d_beta_v, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_m, d_beta_rv, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_m, d_beta_a, d_alpha, d_phi_rv);

  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_m, d_beta_v, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_m, d_beta_rv, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_m, d_beta_a, d_alpha, d_phi_a);

  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_dm1, d_beta_dv, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_dm1, d_beta_drv, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_dm1, d_beta_da, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_dm1, d_beta, d_alpha, d_phi_v);

  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_dm1, d_beta_dv, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_dm1, d_beta_drv, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_dm1, d_beta_da, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_dm1, d_beta, d_alpha, d_phi_rv);

  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_dm1, d_beta_dv, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_dm1, d_beta_drv, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_dm1, d_beta_da, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_dm1, d_beta, d_alpha, d_phi_a);

  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm2, d_beta_v, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm2, d_beta_rv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm2, d_beta_a, d_alpha, d_phi_dv);

  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm2, d_beta_v, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm2, d_beta_rv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm2, d_beta_a, d_alpha, d_phi_drv);

  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm2, d_beta_v, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm2, d_beta_rv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm2, d_beta_a, d_alpha, d_phi_da);

  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm2, d_beta_v, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm2, d_beta_rv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm2, d_beta_a, d_alpha, d_phi);

  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm3, d_beta_dv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm3, d_beta_drv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm3, d_beta_da, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm3, d_beta, d_alpha, d_phi_dv);

  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm3, d_beta_dv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm3, d_beta_drv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm3, d_beta_da, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm3, d_beta, d_alpha, d_phi_drv);

  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm3, d_beta_dv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm3, d_beta_drv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm3, d_beta_da, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm3, d_beta, d_alpha, d_phi_da);

  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm3, d_beta_dv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm3, d_beta_drv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm3, d_beta_da, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dm3, d_beta, d_alpha, d_phi);

  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_v, d_beta_dv, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_v, d_beta_drv, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_v, d_beta_da, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_v, d_beta, d_alpha, d_phi_v);

  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_v, d_beta_dv, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_v, d_beta_drv, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_v, d_beta_da, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_v, d_beta, d_alpha, d_phi_rv);

  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_v, d_beta_dv, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_v, d_beta_drv, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_v, d_beta_da, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_a| d_x_v, d_beta, d_alpha, d_phi_a);

  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dv, d_beta_dv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dv, d_beta_drv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dv, d_beta_da, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dv, d_beta, d_alpha, d_phi_dv);

  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dv, d_beta_dv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dv, d_beta_drv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dv, d_beta_da, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dv, d_beta, d_alpha, d_phi_drv);

  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dv, d_beta_dv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dv, d_beta_drv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dv, d_beta_da, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dv, d_beta, d_alpha, d_phi_da);

  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dv, d_beta_dv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dv, d_beta_drv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dv, d_beta_da, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_dv, d_beta, d_alpha, d_phi);  

  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_rv, d_beta_v, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_rv, d_beta_rv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_rv, d_beta_a, d_alpha, d_phi_dv);

  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_rv, d_beta_v, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_rv, d_beta_rv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_rv, d_beta_a, d_alpha, d_phi_drv);

  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_rv, d_beta_v, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_rv, d_beta_rv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_rv, d_beta_a, d_alpha, d_phi_da);

  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_rv, d_beta_v, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_rv, d_beta_rv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_rv, d_beta_a, d_alpha, d_phi);

  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_drv, d_beta_dv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_drv, d_beta_drv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_drv, d_beta_da, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_drv, d_beta, d_alpha, d_phi_dv);

  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_drv, d_beta_dv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_drv, d_beta_drv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_drv, d_beta_da, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_drv, d_beta, d_alpha, d_phi_drv);

  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_drv, d_beta_dv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_drv, d_beta_drv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_drv, d_beta_da, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_drv, d_beta, d_alpha, d_phi_da);

  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_drv, d_beta_dv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_drv, d_beta_drv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_drv, d_beta_da, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y| d_x_drv, d_beta, d_alpha, d_phi);

  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm2, d_beta_v, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm2, d_beta_rv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm2, d_beta_a, d_alpha, d_phi_dv);

  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm2, d_beta_v, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm2, d_beta_rv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm2, d_beta_a, d_alpha, d_phi_drv);

  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm2, d_beta_v, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm2, d_beta_rv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm2, d_beta_a, d_alpha, d_phi_da);

  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm2, d_beta_v, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm2, d_beta_rv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm2, d_beta_a, d_alpha, d_phi);

  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm3, d_beta_dv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm3, d_beta_drv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm3, d_beta_da, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm3, d_beta, d_alpha, d_phi_dv);

  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm3, d_beta_dv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm3, d_beta_drv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm3, d_beta_da, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm3, d_beta, d_alpha, d_phi_drv);

  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm3, d_beta_dv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm3, d_beta_drv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm3, d_beta_da, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm3, d_beta, d_alpha, d_phi_da);

  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm3, d_beta_dv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm3, d_beta_drv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm3, d_beta_da, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dm3, d_beta, d_alpha, d_phi);

  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dv, d_beta_dv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dv, d_beta_drv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dv, d_beta_da, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dv, d_beta, d_alpha, d_phi_dv);

  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dv, d_beta_dv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dv, d_beta_drv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dv, d_beta_da, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dv, d_beta, d_alpha, d_phi_drv);

  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dv, d_beta_dv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dv, d_beta_drv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dv, d_beta_da, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dv, d_beta, d_alpha, d_phi_da);

  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dv, d_beta_dv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dv, d_beta_drv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dv, d_beta_da, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_dv, d_beta, d_alpha, d_phi);  

  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_rv, d_beta_v, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_rv, d_beta_rv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_rv, d_beta_a, d_alpha, d_phi_dv);

  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_rv, d_beta_v, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_rv, d_beta_rv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_rv, d_beta_a, d_alpha, d_phi_drv);

  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_rv, d_beta_v, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_rv, d_beta_rv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_rv, d_beta_a, d_alpha, d_phi_da);

  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_rv, d_beta_v, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_rv, d_beta_rv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_rv, d_beta_a, d_alpha, d_phi);

  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_drv, d_beta_dv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_drv, d_beta_drv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_drv, d_beta_da, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_drv, d_beta, d_alpha, d_phi_dv);

  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_drv, d_beta_dv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_drv, d_beta_drv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_drv, d_beta_da, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_drv, d_beta, d_alpha, d_phi_drv);

  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_drv, d_beta_dv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_drv, d_beta_drv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_drv, d_beta_da, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_drv, d_beta, d_alpha, d_phi_da);

  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_drv, d_beta_dv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_drv, d_beta_drv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_drv, d_beta_da, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_da| d_x_drv, d_beta, d_alpha, d_phi);

  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_m, d_beta_v, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_m, d_beta_rv, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_m, d_beta_a, d_alpha, d_phi_v);

  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_m, d_beta_v, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_m, d_beta_rv, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_m, d_beta_a, d_alpha, d_phi_rv);

  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_m, d_beta_v, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_m, d_beta_rv, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_m, d_beta_a, d_alpha, d_phi_a);

  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_dm1, d_beta_dv, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_dm1, d_beta_drv, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_dm1, d_beta_da, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_dm1, d_beta, d_alpha, d_phi_v);

  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_dm1, d_beta_dv, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_dm1, d_beta_drv, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_dm1, d_beta_da, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_dm1, d_beta, d_alpha, d_phi_rv);

  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_dm1, d_beta_dv, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_dm1, d_beta_drv, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_dm1, d_beta_da, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_dm1, d_beta, d_alpha, d_phi_a);

  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm2, d_beta_v, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm2, d_beta_rv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm2, d_beta_a, d_alpha, d_phi_dv);

  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm2, d_beta_v, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm2, d_beta_rv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm2, d_beta_a, d_alpha, d_phi_drv);

  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm2, d_beta_v, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm2, d_beta_rv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm2, d_beta_a, d_alpha, d_phi_da);

  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm2, d_beta_v, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm2, d_beta_rv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm2, d_beta_a, d_alpha, d_phi);

  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm3, d_beta_dv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm3, d_beta_drv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm3, d_beta_da, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm3, d_beta, d_alpha, d_phi_dv);

  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm3, d_beta_dv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm3, d_beta_drv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm3, d_beta_da, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm3, d_beta, d_alpha, d_phi_drv);

  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm3, d_beta_dv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm3, d_beta_drv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm3, d_beta_da, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm3, d_beta, d_alpha, d_phi_da);

  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm3, d_beta_dv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm3, d_beta_drv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm3, d_beta_da, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dm3, d_beta, d_alpha, d_phi);

  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_v, d_beta_dv, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_v, d_beta_drv, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_v, d_beta_da, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_v, d_beta, d_alpha, d_phi_v);

  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_v, d_beta_dv, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_v, d_beta_drv, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_v, d_beta_da, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_v, d_beta, d_alpha, d_phi_rv);

  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_v, d_beta_dv, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_v, d_beta_drv, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_v, d_beta_da, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_v| d_x_v, d_beta, d_alpha, d_phi_a);

  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dv, d_beta_dv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dv, d_beta_drv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dv, d_beta_da, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dv, d_beta, d_alpha, d_phi_dv);

  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dv, d_beta_dv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dv, d_beta_drv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dv, d_beta_da, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dv, d_beta, d_alpha, d_phi_drv);

  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dv, d_beta_dv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dv, d_beta_drv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dv, d_beta_da, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dv, d_beta, d_alpha, d_phi_da);

  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dv, d_beta_dv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dv, d_beta_drv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dv, d_beta_da, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_dv, d_beta, d_alpha, d_phi);  

  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_rv, d_beta_v, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_rv, d_beta_rv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_rv, d_beta_a, d_alpha, d_phi_dv);

  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_rv, d_beta_v, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_rv, d_beta_rv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_rv, d_beta_a, d_alpha, d_phi_drv);

  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_rv, d_beta_v, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_rv, d_beta_rv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_rv, d_beta_a, d_alpha, d_phi_da);

  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_rv, d_beta_v, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_rv, d_beta_rv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_rv, d_beta_a, d_alpha, d_phi);

  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_drv, d_beta_dv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_drv, d_beta_drv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_drv, d_beta_da, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_drv, d_beta, d_alpha, d_phi_dv);

  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_drv, d_beta_dv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_drv, d_beta_drv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_drv, d_beta_da, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_drv, d_beta, d_alpha, d_phi_drv);

  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_drv, d_beta_dv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_drv, d_beta_drv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_drv, d_beta_da, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_drv, d_beta, d_alpha, d_phi_da);

  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_drv, d_beta_dv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_drv, d_beta_drv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_drv, d_beta_da, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_dv| d_x_drv, d_beta, d_alpha, d_phi);

  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_m, d_beta_v, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_m, d_beta_rv, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_m, d_beta_a, d_alpha, d_phi_v);

  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_m, d_beta_v, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_m, d_beta_rv, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_m, d_beta_a, d_alpha, d_phi_rv);

  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_m, d_beta_v, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_m, d_beta_rv, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_m, d_beta_a, d_alpha, d_phi_a);

  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_dm1, d_beta_dv, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_dm1, d_beta_drv, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_dm1, d_beta_da, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_dm1, d_beta, d_alpha, d_phi_v);

  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_dm1, d_beta_dv, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_dm1, d_beta_drv, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_dm1, d_beta_da, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_dm1, d_beta, d_alpha, d_phi_rv);

  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_dm1, d_beta_dv, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_dm1, d_beta_drv, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_dm1, d_beta_da, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_dm1, d_beta, d_alpha, d_phi_a);

  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm2, d_beta_v, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm2, d_beta_rv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm2, d_beta_a, d_alpha, d_phi_dv);

  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm2, d_beta_v, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm2, d_beta_rv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm2, d_beta_a, d_alpha, d_phi_drv);

  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm2, d_beta_v, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm2, d_beta_rv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm2, d_beta_a, d_alpha, d_phi_da);

  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm2, d_beta_v, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm2, d_beta_rv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm2, d_beta_a, d_alpha, d_phi);

  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm3, d_beta_dv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm3, d_beta_drv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm3, d_beta_da, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm3, d_beta, d_alpha, d_phi_dv);

  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm3, d_beta_dv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm3, d_beta_drv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm3, d_beta_da, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm3, d_beta, d_alpha, d_phi_drv);

  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm3, d_beta_dv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm3, d_beta_drv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm3, d_beta_da, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm3, d_beta, d_alpha, d_phi_da);

  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm3, d_beta_dv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm3, d_beta_drv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm3, d_beta_da, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dm3, d_beta, d_alpha, d_phi);

  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_v, d_beta_dv, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_v, d_beta_drv, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_v, d_beta_da, d_alpha, d_phi_v);
  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_v, d_beta, d_alpha, d_phi_v);

  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_v, d_beta_dv, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_v, d_beta_drv, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_v, d_beta_da, d_alpha, d_phi_rv);
  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_v, d_beta, d_alpha, d_phi_rv);

  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_v, d_beta_dv, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_v, d_beta_drv, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_v, d_beta_da, d_alpha, d_phi_a);
  transformed_data_real = normal_id_glm_lpdf(d_y_rv| d_x_v, d_beta, d_alpha, d_phi_a);

  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dv, d_beta_dv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dv, d_beta_drv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dv, d_beta_da, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dv, d_beta, d_alpha, d_phi_dv);

  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dv, d_beta_dv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dv, d_beta_drv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dv, d_beta_da, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dv, d_beta, d_alpha, d_phi_drv);

  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dv, d_beta_dv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dv, d_beta_drv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dv, d_beta_da, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dv, d_beta, d_alpha, d_phi_da);

  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dv, d_beta_dv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dv, d_beta_drv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dv, d_beta_da, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_dv, d_beta, d_alpha, d_phi);  

  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_rv, d_beta_v, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_rv, d_beta_rv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_rv, d_beta_a, d_alpha, d_phi_dv);

  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_rv, d_beta_v, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_rv, d_beta_rv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_rv, d_beta_a, d_alpha, d_phi_drv);

  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_rv, d_beta_v, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_rv, d_beta_rv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_rv, d_beta_a, d_alpha, d_phi_da);

  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_rv, d_beta_v, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_rv, d_beta_rv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_rv, d_beta_a, d_alpha, d_phi);

  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_drv, d_beta_dv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_drv, d_beta_drv, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_drv, d_beta_da, d_alpha, d_phi_dv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_drv, d_beta, d_alpha, d_phi_dv);

  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_drv, d_beta_dv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_drv, d_beta_drv, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_drv, d_beta_da, d_alpha, d_phi_drv);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_drv, d_beta, d_alpha, d_phi_drv);

  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_drv, d_beta_dv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_drv, d_beta_drv, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_drv, d_beta_da, d_alpha, d_phi_da);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_drv, d_beta, d_alpha, d_phi_da);

  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_drv, d_beta_dv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_drv, d_beta_drv, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_drv, d_beta_da, d_alpha, d_phi);
  transformed_data_real = normal_id_glm_lpdf(d_y_drv| d_x_drv, d_beta, d_alpha, d_phi);
}
parameters {
  real p_y;
  real p_y_a[N];
  real p_y_da[1];
  vector[N] p_y_v;
  vector[1] p_y_dv;
  row_vector[N] p_y_rv;
  row_vector[1] p_y_drv;

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

  vector<lower=0>[N] p_phi_v;
  vector<lower=0>[1] p_phi_dv;
  row_vector<lower=0>[N] p_phi_rv;
  row_vector<lower=0>[1] p_phi_drv;
  real<lower=0> p_phi_a[N];
  real<lower=0> p_phi_da[1];
  real<lower=0> p_phi;

  real y_p;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_m, p_beta_v, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_m, p_beta_rv, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_m, p_beta_a, p_alpha, p_phi_v);

  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_m, p_beta_v, p_alpha, p_phi_rv);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_m, p_beta_rv, p_alpha, p_phi_rv);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_m, p_beta_a, p_alpha, p_phi_rv);

  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_m, p_beta_v, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_m, p_beta_rv, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_m, p_beta_a, p_alpha, p_phi_a);

  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_dm1, p_beta_dv, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_dm1, p_beta_drv, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_dm1, p_beta_da, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_dm1, p_beta, p_alpha, p_phi_v);

  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_dm1, p_beta_dv, p_alpha, p_phi_rv);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_dm1, p_beta_drv, p_alpha, p_phi_rv);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_dm1, p_beta_da, p_alpha, p_phi_rv);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_dm1, p_beta, p_alpha, p_phi_rv);

  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_dm1, p_beta_dv, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_dm1, p_beta_drv, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_dm1, p_beta_da, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_dm1, p_beta, p_alpha, p_phi_a);

  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm2, p_beta_v, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm2, p_beta_rv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm2, p_beta_a, p_alpha, p_phi_dv);

  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm2, p_beta_v, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm2, p_beta_rv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm2, p_beta_a, p_alpha, p_phi_drv);

  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm2, p_beta_v, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm2, p_beta_rv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm2, p_beta_a, p_alpha, p_phi_da);

  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm2, p_beta_v, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm2, p_beta_rv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm2, p_beta_a, p_alpha, p_phi);

  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm3, p_beta_dv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm3, p_beta_drv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm3, p_beta_da, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm3, p_beta, p_alpha, p_phi_dv);

  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm3, p_beta_dv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm3, p_beta_drv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm3, p_beta_da, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm3, p_beta, p_alpha, p_phi_drv);

  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm3, p_beta_dv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm3, p_beta_drv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm3, p_beta_da, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm3, p_beta, p_alpha, p_phi_da);

  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm3, p_beta_dv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm3, p_beta_drv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm3, p_beta_da, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dm3, p_beta, p_alpha, p_phi);

  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_v, p_beta_dv, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_v, p_beta_drv, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_v, p_beta_da, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_v, p_beta, p_alpha, p_phi_v);

  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_v, p_beta_dv, p_alpha, p_phi_rv);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_v, p_beta_drv, p_alpha, p_phi_rv);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_v, p_beta_da, p_alpha, p_phi_rv);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_v, p_beta, p_alpha, p_phi_rv);

  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_v, p_beta_dv, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_v, p_beta_drv, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_v, p_beta_da, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_a| p_x_v, p_beta, p_alpha, p_phi_a);

  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dv, p_beta_dv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dv, p_beta_drv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dv, p_beta_da, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dv, p_beta, p_alpha, p_phi_dv);

  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dv, p_beta_dv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dv, p_beta_drv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dv, p_beta_da, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dv, p_beta, p_alpha, p_phi_drv);

  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dv, p_beta_dv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dv, p_beta_drv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dv, p_beta_da, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dv, p_beta, p_alpha, p_phi_da);

  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dv, p_beta_dv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dv, p_beta_drv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dv, p_beta_da, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_dv, p_beta, p_alpha, p_phi);  

  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_rv, p_beta_v, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_rv, p_beta_rv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_rv, p_beta_a, p_alpha, p_phi_dv);

  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_rv, p_beta_v, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_rv, p_beta_rv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_rv, p_beta_a, p_alpha, p_phi_drv);

  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_rv, p_beta_v, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_rv, p_beta_rv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_rv, p_beta_a, p_alpha, p_phi_da);

  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_rv, p_beta_v, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_rv, p_beta_rv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_rv, p_beta_a, p_alpha, p_phi);

  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_drv, p_beta_dv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_drv, p_beta_drv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_drv, p_beta_da, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_drv, p_beta, p_alpha, p_phi_dv);

  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_drv, p_beta_dv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_drv, p_beta_drv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_drv, p_beta_da, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_drv, p_beta, p_alpha, p_phi_drv);

  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_drv, p_beta_dv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_drv, p_beta_drv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_drv, p_beta_da, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_drv, p_beta, p_alpha, p_phi_da);

  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_drv, p_beta_dv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_drv, p_beta_drv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_drv, p_beta_da, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y| p_x_drv, p_beta, p_alpha, p_phi);

  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm2, p_beta_v, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm2, p_beta_rv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm2, p_beta_a, p_alpha, p_phi_dv);

  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm2, p_beta_v, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm2, p_beta_rv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm2, p_beta_a, p_alpha, p_phi_drv);

  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm2, p_beta_v, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm2, p_beta_rv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm2, p_beta_a, p_alpha, p_phi_da);

  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm2, p_beta_v, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm2, p_beta_rv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm2, p_beta_a, p_alpha, p_phi);

  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm3, p_beta_dv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm3, p_beta_drv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm3, p_beta_da, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm3, p_beta, p_alpha, p_phi_dv);

  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm3, p_beta_dv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm3, p_beta_drv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm3, p_beta_da, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm3, p_beta, p_alpha, p_phi_drv);

  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm3, p_beta_dv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm3, p_beta_drv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm3, p_beta_da, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm3, p_beta, p_alpha, p_phi_da);

  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm3, p_beta_dv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm3, p_beta_drv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm3, p_beta_da, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dm3, p_beta, p_alpha, p_phi);

  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dv, p_beta_dv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dv, p_beta_drv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dv, p_beta_da, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dv, p_beta, p_alpha, p_phi_dv);

  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dv, p_beta_dv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dv, p_beta_drv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dv, p_beta_da, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dv, p_beta, p_alpha, p_phi_drv);

  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dv, p_beta_dv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dv, p_beta_drv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dv, p_beta_da, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dv, p_beta, p_alpha, p_phi_da);

  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dv, p_beta_dv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dv, p_beta_drv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dv, p_beta_da, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_dv, p_beta, p_alpha, p_phi);  

  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_rv, p_beta_v, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_rv, p_beta_rv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_rv, p_beta_a, p_alpha, p_phi_dv);

  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_rv, p_beta_v, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_rv, p_beta_rv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_rv, p_beta_a, p_alpha, p_phi_drv);

  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_rv, p_beta_v, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_rv, p_beta_rv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_rv, p_beta_a, p_alpha, p_phi_da);

  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_rv, p_beta_v, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_rv, p_beta_rv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_rv, p_beta_a, p_alpha, p_phi);

  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_drv, p_beta_dv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_drv, p_beta_drv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_drv, p_beta_da, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_drv, p_beta, p_alpha, p_phi_dv);

  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_drv, p_beta_dv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_drv, p_beta_drv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_drv, p_beta_da, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_drv, p_beta, p_alpha, p_phi_drv);

  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_drv, p_beta_dv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_drv, p_beta_drv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_drv, p_beta_da, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_drv, p_beta, p_alpha, p_phi_da);

  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_drv, p_beta_dv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_drv, p_beta_drv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_drv, p_beta_da, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_da| p_x_drv, p_beta, p_alpha, p_phi);

  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_m, p_beta_v, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_m, p_beta_rv, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_m, p_beta_a, p_alpha, p_phi_v);

  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_m, p_beta_v, p_alpha, p_phi_rv);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_m, p_beta_rv, p_alpha, p_phi_rv);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_m, p_beta_a, p_alpha, p_phi_rv);

  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_m, p_beta_v, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_m, p_beta_rv, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_m, p_beta_a, p_alpha, p_phi_a);

  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_dm1, p_beta_dv, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_dm1, p_beta_drv, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_dm1, p_beta_da, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_dm1, p_beta, p_alpha, p_phi_v);

  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_dm1, p_beta_dv, p_alpha, p_phi_rv);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_dm1, p_beta_drv, p_alpha, p_phi_rv);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_dm1, p_beta_da, p_alpha, p_phi_rv);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_dm1, p_beta, p_alpha, p_phi_rv);

  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_dm1, p_beta_dv, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_dm1, p_beta_drv, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_dm1, p_beta_da, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_dm1, p_beta, p_alpha, p_phi_a);

  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm2, p_beta_v, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm2, p_beta_rv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm2, p_beta_a, p_alpha, p_phi_dv);

  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm2, p_beta_v, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm2, p_beta_rv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm2, p_beta_a, p_alpha, p_phi_drv);

  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm2, p_beta_v, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm2, p_beta_rv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm2, p_beta_a, p_alpha, p_phi_da);

  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm2, p_beta_v, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm2, p_beta_rv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm2, p_beta_a, p_alpha, p_phi);

  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm3, p_beta_dv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm3, p_beta_drv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm3, p_beta_da, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm3, p_beta, p_alpha, p_phi_dv);

  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm3, p_beta_dv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm3, p_beta_drv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm3, p_beta_da, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm3, p_beta, p_alpha, p_phi_drv);

  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm3, p_beta_dv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm3, p_beta_drv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm3, p_beta_da, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm3, p_beta, p_alpha, p_phi_da);

  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm3, p_beta_dv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm3, p_beta_drv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm3, p_beta_da, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dm3, p_beta, p_alpha, p_phi);

  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_v, p_beta_dv, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_v, p_beta_drv, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_v, p_beta_da, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_v, p_beta, p_alpha, p_phi_v);

  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_v, p_beta_dv, p_alpha, p_phi_rv);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_v, p_beta_drv, p_alpha, p_phi_rv);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_v, p_beta_da, p_alpha, p_phi_rv);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_v, p_beta, p_alpha, p_phi_rv);

  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_v, p_beta_dv, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_v, p_beta_drv, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_v, p_beta_da, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_v| p_x_v, p_beta, p_alpha, p_phi_a);

  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dv, p_beta_dv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dv, p_beta_drv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dv, p_beta_da, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dv, p_beta, p_alpha, p_phi_dv);

  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dv, p_beta_dv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dv, p_beta_drv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dv, p_beta_da, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dv, p_beta, p_alpha, p_phi_drv);

  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dv, p_beta_dv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dv, p_beta_drv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dv, p_beta_da, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dv, p_beta, p_alpha, p_phi_da);

  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dv, p_beta_dv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dv, p_beta_drv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dv, p_beta_da, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_dv, p_beta, p_alpha, p_phi);  

  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_rv, p_beta_v, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_rv, p_beta_rv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_rv, p_beta_a, p_alpha, p_phi_dv);

  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_rv, p_beta_v, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_rv, p_beta_rv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_rv, p_beta_a, p_alpha, p_phi_drv);

  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_rv, p_beta_v, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_rv, p_beta_rv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_rv, p_beta_a, p_alpha, p_phi_da);

  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_rv, p_beta_v, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_rv, p_beta_rv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_rv, p_beta_a, p_alpha, p_phi);

  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_drv, p_beta_dv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_drv, p_beta_drv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_drv, p_beta_da, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_drv, p_beta, p_alpha, p_phi_dv);

  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_drv, p_beta_dv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_drv, p_beta_drv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_drv, p_beta_da, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_drv, p_beta, p_alpha, p_phi_drv);

  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_drv, p_beta_dv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_drv, p_beta_drv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_drv, p_beta_da, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_drv, p_beta, p_alpha, p_phi_da);

  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_drv, p_beta_dv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_drv, p_beta_drv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_drv, p_beta_da, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_dv| p_x_drv, p_beta, p_alpha, p_phi);

  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_m, p_beta_v, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_m, p_beta_rv, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_m, p_beta_a, p_alpha, p_phi_v);

  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_m, p_beta_v, p_alpha, p_phi_rv);
  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_m, p_beta_rv, p_alpha, p_phi_rv);
  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_m, p_beta_a, p_alpha, p_phi_rv);

  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_m, p_beta_v, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_m, p_beta_rv, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_m, p_beta_a, p_alpha, p_phi_a);

  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_dm1, p_beta_dv, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_dm1, p_beta_drv, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_dm1, p_beta_da, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_dm1, p_beta, p_alpha, p_phi_v);

  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_dm1, p_beta_dv, p_alpha, p_phi_rv);
  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_dm1, p_beta_drv, p_alpha, p_phi_rv);
  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_dm1, p_beta_da, p_alpha, p_phi_rv);
  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_dm1, p_beta, p_alpha, p_phi_rv);

  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_dm1, p_beta_dv, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_dm1, p_beta_drv, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_dm1, p_beta_da, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_dm1, p_beta, p_alpha, p_phi_a);

  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm2, p_beta_v, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm2, p_beta_rv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm2, p_beta_a, p_alpha, p_phi_dv);

  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm2, p_beta_v, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm2, p_beta_rv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm2, p_beta_a, p_alpha, p_phi_drv);

  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm2, p_beta_v, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm2, p_beta_rv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm2, p_beta_a, p_alpha, p_phi_da);

  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm2, p_beta_v, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm2, p_beta_rv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm2, p_beta_a, p_alpha, p_phi);

  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm3, p_beta_dv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm3, p_beta_drv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm3, p_beta_da, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm3, p_beta, p_alpha, p_phi_dv);

  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm3, p_beta_dv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm3, p_beta_drv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm3, p_beta_da, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm3, p_beta, p_alpha, p_phi_drv);

  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm3, p_beta_dv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm3, p_beta_drv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm3, p_beta_da, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm3, p_beta, p_alpha, p_phi_da);

  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm3, p_beta_dv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm3, p_beta_drv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm3, p_beta_da, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dm3, p_beta, p_alpha, p_phi);

  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_v, p_beta_dv, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_v, p_beta_drv, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_v, p_beta_da, p_alpha, p_phi_v);
  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_v, p_beta, p_alpha, p_phi_v);

  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_v, p_beta_dv, p_alpha, p_phi_rv);
  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_v, p_beta_drv, p_alpha, p_phi_rv);
  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_v, p_beta_da, p_alpha, p_phi_rv);
  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_v, p_beta, p_alpha, p_phi_rv);

  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_v, p_beta_dv, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_v, p_beta_drv, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_v, p_beta_da, p_alpha, p_phi_a);
  transformed_param_real = normal_id_glm_lpdf(p_y_rv| p_x_v, p_beta, p_alpha, p_phi_a);

  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dv, p_beta_dv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dv, p_beta_drv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dv, p_beta_da, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dv, p_beta, p_alpha, p_phi_dv);

  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dv, p_beta_dv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dv, p_beta_drv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dv, p_beta_da, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dv, p_beta, p_alpha, p_phi_drv);

  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dv, p_beta_dv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dv, p_beta_drv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dv, p_beta_da, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dv, p_beta, p_alpha, p_phi_da);

  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dv, p_beta_dv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dv, p_beta_drv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dv, p_beta_da, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_dv, p_beta, p_alpha, p_phi);  

  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_rv, p_beta_v, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_rv, p_beta_rv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_rv, p_beta_a, p_alpha, p_phi_dv);

  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_rv, p_beta_v, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_rv, p_beta_rv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_rv, p_beta_a, p_alpha, p_phi_drv);

  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_rv, p_beta_v, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_rv, p_beta_rv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_rv, p_beta_a, p_alpha, p_phi_da);

  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_rv, p_beta_v, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_rv, p_beta_rv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_rv, p_beta_a, p_alpha, p_phi);

  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_drv, p_beta_dv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_drv, p_beta_drv, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_drv, p_beta_da, p_alpha, p_phi_dv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_drv, p_beta, p_alpha, p_phi_dv);

  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_drv, p_beta_dv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_drv, p_beta_drv, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_drv, p_beta_da, p_alpha, p_phi_drv);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_drv, p_beta, p_alpha, p_phi_drv);

  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_drv, p_beta_dv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_drv, p_beta_drv, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_drv, p_beta_da, p_alpha, p_phi_da);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_drv, p_beta, p_alpha, p_phi_da);

  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_drv, p_beta_dv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_drv, p_beta_drv, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_drv, p_beta_da, p_alpha, p_phi);
  transformed_param_real = normal_id_glm_lpdf(p_y_drv| p_x_drv, p_beta, p_alpha, p_phi);
}
model {  
  y_p ~ normal(0,1); // in case anyone tries to run it
}
