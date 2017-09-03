
data {
  int d_int;
  int d_int_array[d_int];
  real d_real_array[d_int];
  matrix[d_int,d_int] d_matrix_array[d_int];
  vector[d_int] d_vector_array[d_int];
  row_vector[d_int] d_row_vector_array[d_int];
  int d_int_2d_array[d_int, d_int];
  real d_real_2d_array[d_int, d_int];
  matrix[d_int,d_int] d_matrix_2d_array[d_int, d_int];
  vector[d_int] d_vector_2d_array[d_int, d_int];
  row_vector[d_int] d_row_vector_2d_array[d_int, d_int];
  int d_int_3d_array[d_int, d_int, d_int];
  real d_real_3d_array[d_int, d_int, d_int];
  matrix[d_int,d_int] d_matrix_3d_array[d_int, d_int, d_int];
  vector[d_int] d_vector_3d_array[d_int, d_int, d_int];
  row_vector[d_int] d_row_vector_3d_array[d_int, d_int, d_int];
}

transformed data {
  int transformed_data_int_array[d_int];
  int transformed_data_int_array2[2 * d_int];
  real transformed_data_real_array[d_int];
  real transformed_data_real_array2[2 * d_int];
  matrix[d_int,d_int] transformed_data_matrix_array[d_int];
  matrix[d_int,d_int] transformed_data_matrix_array2[2 * d_int];
  vector[d_int] transformed_data_vector_array[d_int];
  vector[d_int] transformed_data_vector_array2[2 * d_int];
  row_vector[d_int] transformed_data_row_vector_array[d_int];
  row_vector[d_int] transformed_data_row_vector_array2[2 * d_int];
  int transformed_data_int_2d_array[d_int, d_int];
  int transformed_data_int_2d_array2[2 * d_int, d_int];
  real transformed_data_real_2d_array[d_int, d_int];
  real transformed_data_real_2d_array2[2 * d_int, d_int];
  matrix[d_int,d_int] transformed_data_matrix_2d_array[d_int, d_int];
  matrix[d_int,d_int] transformed_data_matrix_2d_array2[2 * d_int, d_int];
  vector[d_int] transformed_data_vector_2d_array[d_int, d_int];
  vector[d_int] transformed_data_vector_2d_array2[2 * d_int, d_int];
  row_vector[d_int] transformed_data_row_vector_2d_array[d_int, d_int];
  row_vector[d_int] transformed_data_row_vector_2d_array2[2 * d_int, d_int];
  int transformed_data_int_3d_array[d_int, d_int, d_int];
  int transformed_data_int_3d_array2[2 * d_int, d_int, d_int];
  real transformed_data_real_3d_array[d_int, d_int, d_int];
  real transformed_data_real_3d_array2[2 * d_int, d_int, d_int];
  matrix[d_int,d_int] transformed_data_matrix_3d_array[d_int, d_int, d_int];
  matrix[d_int,d_int] transformed_data_matrix_3d_array2[2 * d_int, d_int, d_int];
  vector[d_int] transformed_data_vector_3d_array[d_int, d_int, d_int];
  vector[d_int] transformed_data_vector_3d_array2[2 * d_int, d_int, d_int];
  row_vector[d_int] transformed_data_row_vector_3d_array[d_int, d_int, d_int];
  row_vector[d_int] transformed_data_row_vector_3d_array2[2 * d_int, d_int, d_int];

  transformed_data_int_array2 = append_array(d_int_array, d_int_array);
  transformed_data_int_array2 = append_array(d_int_array, transformed_data_int_array);
  transformed_data_int_array2 = append_array(transformed_data_int_array, d_int_array);

  transformed_data_real_array2 = append_array(d_real_array, d_real_array);
  transformed_data_real_array2 = append_array(d_real_array, transformed_data_real_array);
  transformed_data_real_array2 = append_array(transformed_data_real_array, d_real_array);

  transformed_data_matrix_array2 = append_array(d_matrix_array, d_matrix_array);
  transformed_data_matrix_array2 = append_array(d_matrix_array, transformed_data_matrix_array);
  transformed_data_matrix_array2 = append_array(transformed_data_matrix_array, d_matrix_array);

  transformed_data_vector_array2 = append_array(d_vector_array, d_vector_array);
  transformed_data_vector_array2 = append_array(d_vector_array, transformed_data_vector_array);
  transformed_data_vector_array2 = append_array(transformed_data_vector_array, d_vector_array);

  transformed_data_row_vector_array2 = append_array(d_row_vector_array, d_row_vector_array);
  transformed_data_row_vector_array2 = append_array(d_row_vector_array, transformed_data_row_vector_array);
  transformed_data_row_vector_array2 = append_array(transformed_data_row_vector_array, d_row_vector_array);

  transformed_data_int_2d_array2 = append_array(d_int_2d_array, d_int_2d_array);
  transformed_data_int_2d_array2 = append_array(d_int_2d_array, transformed_data_int_2d_array);
  transformed_data_int_2d_array2 = append_array(transformed_data_int_2d_array, d_int_2d_array);

  transformed_data_real_2d_array2 = append_array(d_real_2d_array, d_real_2d_array);
  transformed_data_real_2d_array2 = append_array(d_real_2d_array, transformed_data_real_2d_array);
  transformed_data_real_2d_array2 = append_array(transformed_data_real_2d_array, d_real_2d_array);

  transformed_data_matrix_2d_array2 = append_array(d_matrix_2d_array, d_matrix_2d_array);
  transformed_data_matrix_2d_array2 = append_array(d_matrix_2d_array, transformed_data_matrix_2d_array);
  transformed_data_matrix_2d_array2 = append_array(transformed_data_matrix_2d_array, d_matrix_2d_array);

  transformed_data_vector_2d_array2 = append_array(d_vector_2d_array, d_vector_2d_array);
  transformed_data_vector_2d_array2 = append_array(d_vector_2d_array, transformed_data_vector_2d_array);
  transformed_data_vector_2d_array2 = append_array(transformed_data_vector_2d_array, d_vector_2d_array);

  transformed_data_row_vector_2d_array2 = append_array(d_row_vector_2d_array, d_row_vector_2d_array);
  transformed_data_row_vector_2d_array2 = append_array(d_row_vector_2d_array, transformed_data_row_vector_2d_array);
  transformed_data_row_vector_2d_array2 = append_array(transformed_data_row_vector_2d_array, d_row_vector_2d_array);

  transformed_data_int_3d_array2 = append_array(d_int_3d_array, d_int_3d_array);
  transformed_data_int_3d_array2 = append_array(d_int_3d_array, transformed_data_int_3d_array);
  transformed_data_int_3d_array2 = append_array(transformed_data_int_3d_array, d_int_3d_array);

  transformed_data_real_3d_array2 = append_array(d_real_3d_array, d_real_3d_array);
  transformed_data_real_3d_array2 = append_array(d_real_3d_array, transformed_data_real_3d_array);
  transformed_data_real_3d_array2 = append_array(transformed_data_real_3d_array, d_real_3d_array);

  transformed_data_matrix_3d_array2 = append_array(d_matrix_3d_array, d_matrix_3d_array);
  transformed_data_matrix_3d_array2 = append_array(d_matrix_3d_array, transformed_data_matrix_3d_array);
  transformed_data_matrix_3d_array2 = append_array(transformed_data_matrix_3d_array, d_matrix_3d_array);

  transformed_data_vector_3d_array2 = append_array(d_vector_3d_array, d_vector_3d_array);
  transformed_data_vector_3d_array2 = append_array(d_vector_3d_array, transformed_data_vector_3d_array);
  transformed_data_vector_3d_array2 = append_array(transformed_data_vector_3d_array, d_vector_3d_array);

  transformed_data_row_vector_3d_array2 = append_array(d_row_vector_3d_array, d_row_vector_3d_array);
  transformed_data_row_vector_3d_array2 = append_array(d_row_vector_3d_array, transformed_data_row_vector_3d_array);
  transformed_data_row_vector_3d_array2 = append_array(transformed_data_row_vector_3d_array, d_row_vector_3d_array);
}

parameters {
  real p_real;
  real p_real_array[d_int];
  matrix[d_int,d_int] p_matrix_array[d_int];
  vector[d_int] p_vector_array[d_int];
  row_vector[d_int] p_row_vector_array[d_int];
  real p_real_2d_array[d_int, d_int];
  matrix[d_int,d_int] p_matrix_2d_array[d_int, d_int];
  vector[d_int] p_vector_2d_array[d_int, d_int];
  row_vector[d_int] p_row_vector_2d_array[d_int, d_int];
  real p_real_3d_array[d_int, d_int, d_int];
  matrix[d_int,d_int] p_matrix_3d_array[d_int, d_int, d_int];
  vector[d_int] p_vector_3d_array[d_int, d_int, d_int];
  row_vector[d_int] p_row_vector_3d_array[d_int, d_int, d_int];
}

transformed parameters {
  real transformed_param_real_array[d_int];
  real transformed_param_real_array2[2 * d_int];
  matrix[d_int,d_int] transformed_param_matrix_array[d_int];
  matrix[d_int,d_int] transformed_param_matrix_array2[2 * d_int];
  vector[d_int] transformed_param_vector_array[d_int];
  vector[d_int] transformed_param_vector_array2[2 * d_int];
  row_vector[d_int] transformed_param_row_vector_array[d_int];
  row_vector[d_int] transformed_param_row_vector_array2[2 * d_int];
  real transformed_param_real_2d_array[d_int, d_int];
  real transformed_param_real_2d_array2[2 * d_int, d_int];
  matrix[d_int,d_int] transformed_param_matrix_2d_array[d_int, d_int];
  matrix[d_int,d_int] transformed_param_matrix_2d_array2[2 * d_int, d_int];
  vector[d_int] transformed_param_vector_2d_array[d_int, d_int];
  vector[d_int] transformed_param_vector_2d_array2[2 * d_int, d_int];
  row_vector[d_int] transformed_param_row_vector_2d_array[d_int, d_int];
  row_vector[d_int] transformed_param_row_vector_2d_array2[2 * d_int, d_int];
  real transformed_param_real_3d_array[d_int, d_int, d_int];
  real transformed_param_real_3d_array2[2 * d_int, d_int, d_int];
  matrix[d_int,d_int] transformed_param_matrix_3d_array[d_int, d_int, d_int];
  matrix[d_int,d_int] transformed_param_matrix_3d_array2[2 * d_int, d_int, d_int];
  vector[d_int] transformed_param_vector_3d_array[d_int, d_int, d_int];
  vector[d_int] transformed_param_vector_3d_array2[2 * d_int, d_int, d_int];
  row_vector[d_int] transformed_param_row_vector_3d_array[d_int, d_int, d_int];
  row_vector[d_int] transformed_param_row_vector_3d_array2[2 * d_int, d_int, d_int];

  transformed_param_real_array2 = append_array(p_real_array, p_real_array);
  transformed_param_real_array2 = append_array(p_real_array, d_real_array);
  transformed_param_real_array2 = append_array(transformed_param_real_array, p_real_array);
  transformed_param_real_array2 = append_array(transformed_data_real_array, p_real_array);
  transformed_param_real_array2 = append_array(d_real_array, p_real_array);
  transformed_param_real_array2 = append_array(p_real_array, transformed_param_real_array);
  transformed_param_real_array2 = append_array(p_real_array, transformed_data_real_array);

  transformed_param_matrix_array2 = append_array(p_matrix_array, p_matrix_array);
  transformed_param_matrix_array2 = append_array(p_matrix_array, d_matrix_array);
  transformed_param_matrix_array2 = append_array(p_matrix_array, transformed_param_matrix_array);
  transformed_param_matrix_array2 = append_array(p_matrix_array, transformed_data_matrix_array);
  transformed_param_matrix_array2 = append_array(d_matrix_array, p_matrix_array);
  transformed_param_matrix_array2 = append_array(transformed_param_matrix_array, p_matrix_array);
  transformed_param_matrix_array2 = append_array(transformed_data_matrix_array, p_matrix_array);

  transformed_param_vector_array2 = append_array(p_vector_array, p_vector_array);
  transformed_param_vector_array2 = append_array(p_vector_array, d_vector_array);
  transformed_param_vector_array2 = append_array(p_vector_array, transformed_param_vector_array);
  transformed_param_vector_array2 = append_array(p_vector_array, transformed_data_vector_array);
  transformed_param_vector_array2 = append_array(d_vector_array, p_vector_array);
  transformed_param_vector_array2 = append_array(transformed_param_vector_array, p_vector_array);
  transformed_param_vector_array2 = append_array(transformed_data_vector_array, p_vector_array);

  transformed_param_row_vector_array2 = append_array(p_row_vector_array, p_row_vector_array);
  transformed_param_row_vector_array2 = append_array(p_row_vector_array, d_row_vector_array);
  transformed_param_row_vector_array2 = append_array(p_row_vector_array, transformed_param_row_vector_array);
  transformed_param_row_vector_array2 = append_array(p_row_vector_array, transformed_data_row_vector_array);
  transformed_param_row_vector_array2 = append_array(d_row_vector_array, p_row_vector_array);
  transformed_param_row_vector_array2 = append_array(transformed_param_row_vector_array, p_row_vector_array);
  transformed_param_row_vector_array2 = append_array(transformed_data_row_vector_array, p_row_vector_array);

  transformed_param_real_2d_array2 = append_array(p_real_2d_array, p_real_2d_array);
  transformed_param_real_2d_array2 = append_array(p_real_2d_array, d_real_2d_array);
  transformed_param_real_2d_array2 = append_array(transformed_param_real_2d_array, p_real_2d_array);
  transformed_param_real_2d_array2 = append_array(transformed_data_real_2d_array, p_real_2d_array);
  transformed_param_real_2d_array2 = append_array(d_real_2d_array, p_real_2d_array);
  transformed_param_real_2d_array2 = append_array(p_real_2d_array, transformed_param_real_2d_array);
  transformed_param_real_2d_array2 = append_array(p_real_2d_array, transformed_data_real_2d_array);

  transformed_param_matrix_2d_array2 = append_array(p_matrix_2d_array, p_matrix_2d_array);
  transformed_param_matrix_2d_array2 = append_array(p_matrix_2d_array, d_matrix_2d_array);
  transformed_param_matrix_2d_array2 = append_array(p_matrix_2d_array, transformed_param_matrix_2d_array);
  transformed_param_matrix_2d_array2 = append_array(p_matrix_2d_array, transformed_data_matrix_2d_array);
  transformed_param_matrix_2d_array2 = append_array(d_matrix_2d_array, p_matrix_2d_array);
  transformed_param_matrix_2d_array2 = append_array(transformed_param_matrix_2d_array, p_matrix_2d_array);
  transformed_param_matrix_2d_array2 = append_array(transformed_data_matrix_2d_array, p_matrix_2d_array);

  transformed_param_vector_2d_array2 = append_array(p_vector_2d_array, p_vector_2d_array);
  transformed_param_vector_2d_array2 = append_array(p_vector_2d_array, d_vector_2d_array);
  transformed_param_vector_2d_array2 = append_array(p_vector_2d_array, transformed_param_vector_2d_array);
  transformed_param_vector_2d_array2 = append_array(p_vector_2d_array, transformed_data_vector_2d_array);
  transformed_param_vector_2d_array2 = append_array(d_vector_2d_array, p_vector_2d_array);
  transformed_param_vector_2d_array2 = append_array(transformed_param_vector_2d_array, p_vector_2d_array);
  transformed_param_vector_2d_array2 = append_array(transformed_data_vector_2d_array, p_vector_2d_array);

  transformed_param_row_vector_2d_array2 = append_array(p_row_vector_2d_array, p_row_vector_2d_array);
  transformed_param_row_vector_2d_array2 = append_array(p_row_vector_2d_array, d_row_vector_2d_array);
  transformed_param_row_vector_2d_array2 = append_array(p_row_vector_2d_array, transformed_param_row_vector_2d_array);
  transformed_param_row_vector_2d_array2 = append_array(p_row_vector_2d_array, transformed_data_row_vector_2d_array);
  transformed_param_row_vector_2d_array2 = append_array(d_row_vector_2d_array, p_row_vector_2d_array);
  transformed_param_row_vector_2d_array2 = append_array(transformed_param_row_vector_2d_array, p_row_vector_2d_array);
  transformed_param_row_vector_2d_array2 = append_array(transformed_data_row_vector_2d_array, p_row_vector_2d_array);

  transformed_param_real_3d_array2 = append_array(p_real_3d_array, p_real_3d_array);
  transformed_param_real_3d_array2 = append_array(p_real_3d_array, d_real_3d_array);
  transformed_param_real_3d_array2 = append_array(transformed_param_real_3d_array, p_real_3d_array);
  transformed_param_real_3d_array2 = append_array(transformed_data_real_3d_array, p_real_3d_array);
  transformed_param_real_3d_array2 = append_array(d_real_3d_array, p_real_3d_array);
  transformed_param_real_3d_array2 = append_array(p_real_3d_array, transformed_param_real_3d_array);
  transformed_param_real_3d_array2 = append_array(p_real_3d_array, transformed_data_real_3d_array);

  transformed_param_matrix_3d_array2 = append_array(p_matrix_3d_array, p_matrix_3d_array);
  transformed_param_matrix_3d_array2 = append_array(p_matrix_3d_array, d_matrix_3d_array);
  transformed_param_matrix_3d_array2 = append_array(p_matrix_3d_array, transformed_param_matrix_3d_array);
  transformed_param_matrix_3d_array2 = append_array(p_matrix_3d_array, transformed_data_matrix_3d_array);
  transformed_param_matrix_3d_array2 = append_array(d_matrix_3d_array, p_matrix_3d_array);
  transformed_param_matrix_3d_array2 = append_array(transformed_param_matrix_3d_array, p_matrix_3d_array);
  transformed_param_matrix_3d_array2 = append_array(transformed_data_matrix_3d_array, p_matrix_3d_array);

  transformed_param_vector_3d_array2 = append_array(p_vector_3d_array, p_vector_3d_array);
  transformed_param_vector_3d_array2 = append_array(p_vector_3d_array, d_vector_3d_array);
  transformed_param_vector_3d_array2 = append_array(p_vector_3d_array, transformed_param_vector_3d_array);
  transformed_param_vector_3d_array2 = append_array(p_vector_3d_array, transformed_data_vector_3d_array);
  transformed_param_vector_3d_array2 = append_array(d_vector_3d_array, p_vector_3d_array);
  transformed_param_vector_3d_array2 = append_array(transformed_param_vector_3d_array, p_vector_3d_array);
  transformed_param_vector_3d_array2 = append_array(transformed_data_vector_3d_array, p_vector_3d_array);

  transformed_param_row_vector_3d_array2 = append_array(p_row_vector_3d_array, p_row_vector_3d_array);
  transformed_param_row_vector_3d_array2 = append_array(p_row_vector_3d_array, d_row_vector_3d_array);
  transformed_param_row_vector_3d_array2 = append_array(p_row_vector_3d_array, transformed_param_row_vector_3d_array);
  transformed_param_row_vector_3d_array2 = append_array(p_row_vector_3d_array, transformed_data_row_vector_3d_array);
  transformed_param_row_vector_3d_array2 = append_array(d_row_vector_3d_array, p_row_vector_3d_array);
  transformed_param_row_vector_3d_array2 = append_array(transformed_param_row_vector_3d_array, p_row_vector_3d_array);
  transformed_param_row_vector_3d_array2 = append_array(transformed_data_row_vector_3d_array, p_row_vector_3d_array);
}

model {
  p_real ~ normal(0,1);
}
