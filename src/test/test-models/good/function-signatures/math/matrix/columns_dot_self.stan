data { 
  int d_int;
  matrix[d_int,d_int] d_matrix;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}
transformed data {
  row_vector[d_int] transformed_data_row_vector;

  transformed_data_row_vector <- columns_dot_self(d_vector);
  transformed_data_row_vector <- columns_dot_self(d_row_vector);
  transformed_data_row_vector <- columns_dot_self(d_matrix);
}
parameters {
  matrix[d_int,d_int] p_matrix;
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;
  real y_p;
}
transformed parameters {
  row_vector[d_int] transformed_param_row_vector;

  transformed_param_row_vector <- columns_dot_self(d_vector);
  transformed_param_row_vector <- columns_dot_self(d_row_vector);
  transformed_param_row_vector <- columns_dot_self(d_matrix);

  transformed_param_row_vector <- columns_dot_self(p_vector);
  transformed_param_row_vector <- columns_dot_self(p_row_vector);
  transformed_param_row_vector <- columns_dot_self(p_matrix);
}
model {  
  y_p ~ normal(0,1);
}
