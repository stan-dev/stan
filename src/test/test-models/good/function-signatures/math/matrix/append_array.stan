data {
  int d_int;
  real d_real_array[d_int];
  matrix[d_int,d_int] d_matrix[d_int];
  vector[d_int] d_vector[d_int];
  row_vector[d_int] d_row_vector[d_int];
}

transformed data {
  real transformed_data_real_array[d_int];
  real transformed_data_real_array2[2 * d_int];
  matrix[d_int,d_int] transformed_data_matrix[d_int];
  matrix[d_int,d_int] transformed_data_matrix2[2 * d_int];
  vector[d_int] transformed_data_vector[d_int];
  vector[d_int] transformed_data_vector2[2 * d_int];
  row_vector[d_int] transformed_data_row_vector[d_int];
  row_vector[d_int] transformed_data_row_vector2[2 * d_int];

  transformed_data_real_array2 = append_array(d_real_array, d_real_array);
  transformed_data_real_array2 = append_array(d_real_array, transformed_data_real_array);
  transformed_data_real_array2 = append_array(transformed_data_real_array, d_real_array);

  transformed_data_matrix2 = append_array(d_matrix, d_matrix);
  transformed_data_matrix2 = append_array(d_matrix, transformed_data_matrix);
  transformed_data_matrix2 = append_array(transformed_data_matrix, d_matrix);

  transformed_data_vector2 = append_array(d_vector, d_vector);
  transformed_data_vector2 = append_array(d_vector, transformed_data_vector);
  transformed_data_vector2 = append_array(transformed_data_vector, d_vector);

  transformed_data_row_vector2 = append_array(d_row_vector, d_row_vector);
  transformed_data_row_vector2 = append_array(d_row_vector, transformed_data_row_vector);
  transformed_data_row_vector2 = append_array(transformed_data_row_vector, d_row_vector);
}

parameters {
  real p_real;
  real p_real_array[d_int];
  matrix[d_int,d_int] p_matrix[d_int];
  vector[d_int] p_vector[d_int];
  row_vector[d_int] p_row_vector[d_int];
}

transformed parameters {
  real transformed_param_real_array[d_int];
  real transformed_param_real_array2[2 * d_int];
  matrix[d_int,d_int] transformed_param_matrix[d_int];
  matrix[d_int,d_int] transformed_param_matrix2[2 * d_int];
  vector[d_int] transformed_param_vector[d_int];
  vector[d_int] transformed_param_vector2[2 * d_int];
  row_vector[d_int] transformed_param_row_vector[d_int];
  row_vector[d_int] transformed_param_row_vector2[2 * d_int];

  transformed_param_real_array2 = append_array(p_real_array, p_real_array);
  transformed_param_real_array2 = append_array(p_real_array, d_real_array);
  transformed_param_real_array2 = append_array(transformed_param_real_array, p_real_array);
  transformed_param_real_array2 = append_array(transformed_data_real_array, p_real_array);
  transformed_param_real_array2 = append_array(d_real_array, p_real_array);
  transformed_param_real_array2 = append_array(p_real_array, transformed_param_real_array);
  transformed_param_real_array2 = append_array(p_real_array, transformed_data_real_array);

  transformed_param_matrix2 = append_array(p_matrix, p_matrix);
  transformed_param_matrix2 = append_array(p_matrix, d_matrix);
  transformed_param_matrix2 = append_array(p_matrix, transformed_param_matrix);
  transformed_param_matrix2 = append_array(p_matrix, transformed_data_matrix);
  transformed_param_matrix2 = append_array(d_matrix, p_matrix);
  transformed_param_matrix2 = append_array(transformed_param_matrix, p_matrix);
  transformed_param_matrix2 = append_array(transformed_data_matrix, p_matrix);

  transformed_param_vector2 = append_array(p_vector, p_vector);
  transformed_param_vector2 = append_array(p_vector, d_vector);
  transformed_param_vector2 = append_array(p_vector, transformed_param_vector);
  transformed_param_vector2 = append_array(p_vector, transformed_data_vector);
  transformed_param_vector2 = append_array(d_vector, p_vector);
  transformed_param_vector2 = append_array(transformed_param_vector, p_vector);
  transformed_param_vector2 = append_array(transformed_data_vector, p_vector);

  transformed_param_row_vector2 = append_array(p_row_vector, p_row_vector);
  transformed_param_row_vector2 = append_array(p_row_vector, d_row_vector);
  transformed_param_row_vector2 = append_array(p_row_vector, transformed_param_row_vector);
  transformed_param_row_vector2 = append_array(p_row_vector, transformed_data_row_vector);
  transformed_param_row_vector2 = append_array(d_row_vector, p_row_vector);
  transformed_param_row_vector2 = append_array(transformed_param_row_vector, p_row_vector);
  transformed_param_row_vector2 = append_array(transformed_data_row_vector, p_row_vector);
}

model {
  p_real ~ normal(0,1);
}
