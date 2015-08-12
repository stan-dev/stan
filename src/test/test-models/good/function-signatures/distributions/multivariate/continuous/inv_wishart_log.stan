data { 
  int d_int;
  real d_real;
  matrix[d_int,d_int] d_matrix;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}
transformed data {
  real transformed_data_real;

  transformed_data_real <- inv_wishart_log(d_matrix, d_real, d_matrix);
  transformed_data_real <- inv_wishart_log(d_matrix, d_int, d_matrix);
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

  transformed_param_real <- inv_wishart_log(d_matrix, d_real, d_matrix);
  transformed_param_real <- inv_wishart_log(d_matrix, d_real, p_matrix);
  transformed_param_real <- inv_wishart_log(d_matrix, p_real, d_matrix);
  transformed_param_real <- inv_wishart_log(d_matrix, p_real, p_matrix);
  transformed_param_real <- inv_wishart_log(p_matrix, d_real, d_matrix);
  transformed_param_real <- inv_wishart_log(p_matrix, d_real, p_matrix);
  transformed_param_real <- inv_wishart_log(p_matrix, p_real, d_matrix);
  transformed_param_real <- inv_wishart_log(p_matrix, p_real, p_matrix);

  transformed_param_real <- inv_wishart_log(d_matrix, d_int, d_matrix);
  transformed_param_real <- inv_wishart_log(d_matrix, d_int, p_matrix);
  transformed_param_real <- inv_wishart_log(p_matrix, d_int, d_matrix);
  transformed_param_real <- inv_wishart_log(p_matrix, d_int, p_matrix);
}
model {  
  y_p ~ normal(0,1);
}
