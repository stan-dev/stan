transformed data {
  int N = 2;
  int M = 3;
  int d_y = 1;
  int d_y_a[N] = {1, 0};
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

  real transformed_data_real;

  transformed_data_real = bernoulli_logit_glm_lpmf(d_y_a| d_x_v, d_beta_fm, d_alpha);
  transformed_data_real = bernoulli_logit_glm_lpmf(d_y_a| d_x_v, d_beta_fa, d_alpha);
  transformed_data_real = bernoulli_logit_glm_lpmf(d_y_a| d_x_v, d_beta, d_alpha);

  transformed_data_real = bernoulli_logit_glm_lpmf(d_y| d_x_rv, d_beta_v, d_alpha);
  transformed_data_real = bernoulli_logit_glm_lpmf(d_y| d_x_rv, d_beta_rv, d_alpha);
  transformed_data_real = bernoulli_logit_glm_lpmf(d_y| d_x_rv, d_beta_a, d_alpha);

  transformed_data_real = bernoulli_logit_glm_lpmf(d_y_a| d_x_m, d_beta_v, d_alpha);
  transformed_data_real = bernoulli_logit_glm_lpmf(d_y_a| d_x_m, d_beta_rv, d_alpha);
  transformed_data_real = bernoulli_logit_glm_lpmf(d_y_a| d_x_m, d_beta_a, d_alpha);
}
parameters {
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

  real y_p;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real = bernoulli_logit_glm_lpmf(d_y_a| p_x_v, p_beta_fm, p_alpha);
  transformed_param_real = bernoulli_logit_glm_lpmf(d_y_a| p_x_v, p_beta_fa, p_alpha);
  transformed_param_real = bernoulli_logit_glm_lpmf(d_y_a| p_x_v, p_beta, p_alpha);

  transformed_param_real = bernoulli_logit_glm_lpmf(d_y| p_x_rv, p_beta_v, p_alpha);
  transformed_param_real = bernoulli_logit_glm_lpmf(d_y| p_x_rv, p_beta_rv, p_alpha);
  transformed_param_real = bernoulli_logit_glm_lpmf(d_y| p_x_rv, p_beta_a, p_alpha);
  
  transformed_param_real = bernoulli_logit_glm_lpmf(d_y_a| p_x_m, p_beta_v, p_alpha);
  transformed_param_real = bernoulli_logit_glm_lpmf(d_y_a| p_x_m, p_beta_rv, p_alpha);
  transformed_param_real = bernoulli_logit_glm_lpmf(d_y_a| p_x_m, p_beta_a, p_alpha);
}
model {  
  y_p ~ normal(0,1); // in case anyone tries to run it
}
