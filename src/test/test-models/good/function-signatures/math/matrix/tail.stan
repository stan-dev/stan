data { 
  int d_int;
  int d_int_array[d_int];
  int d_int_array_2[d_int, d_int];
  int d_int_array_3[d_int, d_int, d_int];
  real d_real_array[d_int];
  real d_real_array_2[d_int,d_int];
  real d_real_array_3[d_int,d_int,d_int];
  matrix[d_int,d_int] d_matrix_array[d_int];
  matrix[d_int,d_int] d_matrix_array_2[d_int,d_int];
  matrix[d_int,d_int] d_matrix_array_3[d_int,d_int,d_int];
  vector[d_int] d_vector;
  vector[d_int] d_vector_array[d_int];
  vector[d_int] d_vector_array_2[d_int,d_int];
  vector[d_int] d_vector_array_3[d_int,d_int,d_int];
  row_vector[d_int] d_row_vector;
  row_vector[d_int] d_row_vector_array[d_int];
  row_vector[d_int] d_row_vector_array_2[d_int,d_int];
  row_vector[d_int] d_row_vector_array_3[d_int,d_int,d_int];
}

transformed data {
  int transformed_data_int_array[d_int];
  int transformed_data_int_array_2[d_int,d_int];
  int transformed_data_int_array_3[d_int,d_int,d_int];
  real transformed_data_real_array[d_int];
  real transformed_data_real_array_2[d_int,d_int];
  real transformed_data_real_array_3[d_int,d_int,d_int];
  matrix[d_int,d_int] transformed_data_matrix_array[d_int];
  matrix[d_int,d_int] transformed_data_matrix_array_2[d_int,d_int];
  matrix[d_int,d_int] transformed_data_matrix_array_3[d_int,d_int,d_int];
  vector[d_int] transformed_data_vector;
  vector[d_int] transformed_data_vector_array[d_int];
  vector[d_int] transformed_data_vector_array_2[d_int,d_int];
  vector[d_int] transformed_data_vector_array_3[d_int,d_int, d_int];
  row_vector[d_int] transformed_data_row_vector;
  row_vector[d_int] transformed_data_row_vector_array[d_int];
  row_vector[d_int] transformed_data_row_vector_array_2[d_int,d_int];
  row_vector[d_int] transformed_data_row_vector_array_3[d_int,d_int,d_int];

  transformed_data_int_array <- tail(d_int_array,d_int);
  transformed_data_int_array_2 <- tail(d_int_array_2,d_int);
  transformed_data_int_array_3 <- tail(d_int_array_3,d_int);
  transformed_data_real_array <- tail(d_real_array,d_int);
  transformed_data_real_array_2 <- tail(d_real_array_2,d_int);
  transformed_data_real_array_3 <- tail(d_real_array_3,d_int);
  transformed_data_matrix_array <- tail(d_matrix_array,d_int);
  transformed_data_matrix_array_2 <- tail(d_matrix_array_2,d_int);
  transformed_data_matrix_array_3 <- tail(d_matrix_array_3,d_int);
  transformed_data_vector <- tail(d_vector,d_int);
  transformed_data_vector_array <- tail(d_vector_array,d_int);
  transformed_data_vector_array_2 <- tail(d_vector_array_2,d_int);
  transformed_data_vector_array_3 <- tail(d_vector_array_3,d_int);
  transformed_data_row_vector <- tail(d_row_vector,d_int);
  transformed_data_row_vector_array <- tail(d_row_vector_array,d_int);
  transformed_data_row_vector_array_2 <- tail(d_row_vector_array_2,d_int);
  transformed_data_row_vector_array_3 <- tail(d_row_vector_array_3,d_int);
}
parameters {
  real y_p;
  real p_real_array[d_int];
  real p_real_array_2[d_int,d_int];
  real p_real_array_3[d_int,d_int,d_int];
  matrix[d_int,d_int] p_matrix_array[d_int];
  matrix[d_int,d_int] p_matrix_array_2[d_int,d_int];
  matrix[d_int,d_int] p_matrix_array_3[d_int,d_int,d_int];
  vector[d_int] p_vector;
  vector[d_int] p_vector_array[d_int];
  vector[d_int] p_vector_array_2[d_int,d_int];
  vector[d_int] p_vector_array_3[d_int,d_int,d_int];
  row_vector[d_int] p_row_vector;
  row_vector[d_int] p_row_vector_array[d_int];
  row_vector[d_int] p_row_vector_array_2[d_int,d_int];
  row_vector[d_int] p_row_vector_array_3[d_int,d_int,d_int];
}
transformed parameters {
  real transformed_param_real_array[d_int];
  real transformed_param_real_array_2[d_int,d_int];
  real transformed_param_real_array_3[d_int,d_int,d_int];
  matrix[d_int,d_int] transformed_param_matrix_array[d_int];
  matrix[d_int,d_int] transformed_param_matrix_array_2[d_int,d_int];
  matrix[d_int,d_int] transformed_param_matrix_array_3[d_int,d_int,d_int];
  vector[d_int] transformed_param_vector;
  vector[d_int] transformed_param_vector_array[d_int];
  vector[d_int] transformed_param_vector_array_2[d_int,d_int];
  vector[d_int] transformed_param_vector_array_3[d_int,d_int, d_int];
  row_vector[d_int] transformed_param_row_vector;
  row_vector[d_int] transformed_param_row_vector_array[d_int];
  row_vector[d_int] transformed_param_row_vector_array_2[d_int,d_int];
  row_vector[d_int] transformed_param_row_vector_array_3[d_int,d_int,d_int];

  transformed_param_real_array <- tail(d_real_array,d_int);
  transformed_param_real_array_2 <- tail(d_real_array_2,d_int);
  transformed_param_real_array_3 <- tail(d_real_array_3,d_int);
  transformed_param_matrix_array <- tail(d_matrix_array,d_int);
  transformed_param_matrix_array_2 <- tail(d_matrix_array_2,d_int);
  transformed_param_matrix_array_3 <- tail(d_matrix_array_3,d_int);
  transformed_param_vector <- tail(d_vector,d_int);
  transformed_param_vector_array <- tail(d_vector_array,d_int);
  transformed_param_vector_array_2 <- tail(d_vector_array_2,d_int);
  transformed_param_vector_array_3 <- tail(d_vector_array_3,d_int);
  transformed_param_row_vector <- tail(d_row_vector,d_int);
  transformed_param_row_vector_array <- tail(d_row_vector_array,d_int);
  transformed_param_row_vector_array_2 <- tail(d_row_vector_array_2,d_int);
  transformed_param_row_vector_array_3 <- tail(d_row_vector_array_3,d_int);

  transformed_param_real_array <- tail(p_real_array,d_int);
  transformed_param_real_array_2 <- tail(p_real_array_2,d_int);
  transformed_param_real_array_3 <- tail(p_real_array_3,d_int);
  transformed_param_matrix_array <- tail(p_matrix_array,d_int);
  transformed_param_matrix_array_2 <- tail(p_matrix_array_2,d_int);
  transformed_param_matrix_array_3 <- tail(p_matrix_array_3,d_int);
  transformed_param_vector <- tail(p_vector,d_int);
  transformed_param_vector_array <- tail(p_vector_array,d_int);
  transformed_param_vector_array_2 <- tail(p_vector_array_2,d_int);
  transformed_param_vector_array_3 <- tail(p_vector_array_3,d_int);
  transformed_param_row_vector <- tail(p_row_vector,d_int);
  transformed_param_row_vector_array <- tail(p_row_vector_array,d_int);
  transformed_param_row_vector_array_2 <- tail(p_row_vector_array_2,d_int);
  transformed_param_row_vector_array_3 <- tail(p_row_vector_array_3,d_int);
}
model {  
  y_p ~ normal(0,1);
}
