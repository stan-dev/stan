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

  transformed_data_int_array <- head(d_int_array,d_int);
  transformed_data_int_array_2 <- head(d_int_array_2,d_int);
  transformed_data_int_array_3 <- head(d_int_array_3,d_int);
  transformed_data_real_array <- head(d_real_array,d_int);
  transformed_data_real_array_2 <- head(d_real_array_2,d_int);
  transformed_data_real_array_3 <- head(d_real_array_3,d_int);
  transformed_data_matrix_array <- head(d_matrix_array,d_int);
  transformed_data_matrix_array_2 <- head(d_matrix_array_2,d_int);
  transformed_data_matrix_array_3 <- head(d_matrix_array_3,d_int);
  transformed_data_vector <- head(d_vector,d_int);
  transformed_data_vector_array <- head(d_vector_array,d_int);
  transformed_data_vector_array_2 <- head(d_vector_array_2,d_int);
  transformed_data_vector_array_3 <- head(d_vector_array_3,d_int);
  transformed_data_row_vector <- head(d_row_vector,d_int);
  transformed_data_row_vector_array <- head(d_row_vector_array,d_int);
  transformed_data_row_vector_array_2 <- head(d_row_vector_array_2,d_int);
  transformed_data_row_vector_array_3 <- head(d_row_vector_array_3,d_int);
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

  transformed_param_real_array <- head(d_real_array,d_int);
  transformed_param_real_array_2 <- head(d_real_array_2,d_int);
  transformed_param_real_array_3 <- head(d_real_array_3,d_int);
  transformed_param_matrix_array <- head(d_matrix_array,d_int);
  transformed_param_matrix_array_2 <- head(d_matrix_array_2,d_int);
  transformed_param_matrix_array_3 <- head(d_matrix_array_3,d_int);
  transformed_param_vector <- head(d_vector,d_int);
  transformed_param_vector_array <- head(d_vector_array,d_int);
  transformed_param_vector_array_2 <- head(d_vector_array_2,d_int);
  transformed_param_vector_array_3 <- head(d_vector_array_3,d_int);
  transformed_param_row_vector <- head(d_row_vector,d_int);
  transformed_param_row_vector_array <- head(d_row_vector_array,d_int);
  transformed_param_row_vector_array_2 <- head(d_row_vector_array_2,d_int);
  transformed_param_row_vector_array_3 <- head(d_row_vector_array_3,d_int);

  transformed_param_real_array <- head(p_real_array,d_int);
  transformed_param_real_array_2 <- head(p_real_array_2,d_int);
  transformed_param_real_array_3 <- head(p_real_array_3,d_int);
  transformed_param_matrix_array <- head(p_matrix_array,d_int);
  transformed_param_matrix_array_2 <- head(p_matrix_array_2,d_int);
  transformed_param_matrix_array_3 <- head(p_matrix_array_3,d_int);
  transformed_param_vector <- head(p_vector,d_int);
  transformed_param_vector_array <- head(p_vector_array,d_int);
  transformed_param_vector_array_2 <- head(p_vector_array_2,d_int);
  transformed_param_vector_array_3 <- head(p_vector_array_3,d_int);
  transformed_param_row_vector <- head(p_row_vector,d_int);
  transformed_param_row_vector_array <- head(p_row_vector_array,d_int);
  transformed_param_row_vector_array_2 <- head(p_row_vector_array_2,d_int);
  transformed_param_row_vector_array_3 <- head(p_row_vector_array_3,d_int);
}
model {  
  y_p ~ normal(0,1);
}
