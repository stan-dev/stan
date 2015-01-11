data { 
  int d_int;
  matrix[d_int,d_int] d_matrix;
}

transformed data {
  row_vector[d_int] transformed_data_row_vector;

  transformed_data_row_vector <- sub_row(d_matrix, d_int, d_int, d_int);
}
parameters {
  real y_p;
  matrix[d_int,d_int] p_matrix;
}
transformed parameters {
  row_vector[d_int] transformed_param_row_vector;

  transformed_param_row_vector <- sub_row(d_matrix, d_int, d_int, d_int);
  transformed_param_row_vector <- sub_row(p_matrix, d_int, d_int, d_int);
}
model {  
  y_p ~ normal(0,1);
}
