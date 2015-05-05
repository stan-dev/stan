data { 
  int d_int;
  matrix[d_int,d_int] d_matrix;
  vector[d_int] d_vector;
}
transformed data {
  real transformed_data_real;

  transformed_data_real <- multi_gp_log(d_matrix, d_matrix, d_vector);
}
parameters {
  matrix[d_int,d_int] p_matrix;
  vector[d_int] p_vector;
  real y_p;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real <- multi_gp_log(d_matrix, d_matrix, d_vector);
  transformed_param_real <- multi_gp_log(p_matrix, d_matrix, d_vector);
  transformed_param_real <- multi_gp_log(d_matrix, p_matrix, d_vector);
  transformed_param_real <- multi_gp_log(d_matrix, d_matrix, p_vector);
  transformed_param_real <- multi_gp_log(p_matrix, p_matrix, d_vector);
  transformed_param_real <- multi_gp_log(p_matrix, d_matrix, p_vector);
  transformed_param_real <- multi_gp_log(p_matrix, p_matrix, p_vector);
  transformed_param_real <- multi_gp_log(d_matrix, p_matrix, p_vector);
}
model {  
  y_p ~ normal(0,1);
}
