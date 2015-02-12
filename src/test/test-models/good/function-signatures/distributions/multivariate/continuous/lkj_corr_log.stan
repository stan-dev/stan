data { 
  int d_int;
  real d_real;
  matrix[d_int,d_int] d_matrix;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}
transformed data {
  real transformed_data_real;

  transformed_data_real <- lkj_corr_log(d_matrix, d_real);
  transformed_data_real <- lkj_corr_log(d_matrix, d_int);
}
parameters {
  matrix[d_int,d_int] p_matrix;
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;
  real p_real;
  real y_p;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real <- lkj_corr_log(d_matrix, d_real);
  transformed_param_real <- lkj_corr_log(d_matrix, p_real);
  transformed_param_real <- lkj_corr_log(p_matrix, d_real);
  transformed_param_real <- lkj_corr_log(p_matrix, p_real);

  transformed_param_real <- lkj_corr_log(d_matrix, d_int);
  transformed_param_real <- lkj_corr_log(p_matrix, d_int);
}
model {  
  y_p ~ normal(0,1);
}
