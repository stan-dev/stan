data { 
  int d_int;
  matrix[d_int,d_int] d_matrix;
  row_vector[d_int] d_row_vector;
}

transformed data {
  row_vector[d_int] transformed_data_row_vector;
  matrix[d_int,d_int] transformed_data_matrix;

  transformed_data_matrix = mdivide_right(d_matrix,d_matrix);
  transformed_data_row_vector = mdivide_right(d_row_vector,d_matrix);
}
parameters {
  real y_p;
  matrix[d_int,d_int] p_matrix;
  row_vector[d_int] p_row_vector;
}
transformed parameters {
  row_vector[d_int] transformed_param_row_vector;
  matrix[d_int,d_int] transformed_param_matrix;

  transformed_param_matrix = mdivide_right(d_matrix,d_matrix);
  transformed_param_row_vector = mdivide_right(d_row_vector,d_matrix);
  transformed_param_matrix = mdivide_right(p_matrix,d_matrix);
  transformed_param_row_vector = mdivide_right(p_row_vector,d_matrix);
  transformed_param_matrix = mdivide_right(d_matrix,p_matrix);
  transformed_param_row_vector = mdivide_right(d_row_vector,p_matrix);
  transformed_param_matrix = mdivide_right(p_matrix,p_matrix);
  transformed_param_row_vector = mdivide_right(p_row_vector,p_matrix);
}
model {  
  y_p ~ normal(0,1);
}
