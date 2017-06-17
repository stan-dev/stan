data {
  int d_int;
  matrix[d_int,d_int] d_matrix;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
  real d_array[6];
  real d_array2[6, 2];
  int d_iarray[6];
  int d_iarray2[6, 2];
}

transformed data {
  matrix[d_int,d_int] transformed_data_matrix;

  transformed_data_matrix = to_matrix(d_matrix);
  transformed_data_matrix = to_matrix(d_vector);
  transformed_data_matrix = to_matrix(d_row_vector);

  transformed_data_matrix = to_matrix(d_matrix, 4, 2);
  transformed_data_matrix = to_matrix(d_vector, 4, 3);
  transformed_data_matrix = to_matrix(d_row_vector, 5, 2);

  transformed_data_matrix = to_matrix(d_matrix, 4, 2, 1);
  transformed_data_matrix = to_matrix(d_vector, 4, 3, 1);
  transformed_data_matrix = to_matrix(d_row_vector, 5, 2, 4);

  transformed_data_matrix = to_matrix(d_array, 2, 3);
  transformed_data_matrix = to_matrix(d_array, 2, 3, 1);
  transformed_data_matrix = to_matrix(d_iarray, 2, 3);
  transformed_data_matrix = to_matrix(d_iarray, 2, 3, 1);

  transformed_data_matrix = to_matrix(d_array2);
  transformed_data_matrix = to_matrix(d_iarray2);
}
parameters {
  real y_p;
  matrix[d_int,d_int] p_matrix;
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;
  real p_array[6];
  real p_array2[6, 7];
}
transformed parameters {
  matrix[d_int,d_int] transformed_param_matrix;

  transformed_param_matrix = to_matrix(d_matrix);
  transformed_param_matrix = to_matrix(d_vector);
  transformed_param_matrix = to_matrix(d_row_vector);

  transformed_param_matrix = to_matrix(d_matrix, 4, 2);
  transformed_param_matrix = to_matrix(d_vector, 3, 5);
  transformed_param_matrix = to_matrix(d_row_vector, 2, 4);

  transformed_param_matrix = to_matrix(d_matrix, 4, 2, 1);
  transformed_param_matrix = to_matrix(d_vector, 3, 5, 1);
  transformed_param_matrix = to_matrix(d_row_vector, 2, 4, 1);

  transformed_param_matrix = to_matrix(d_array, 3, 2);
  transformed_param_matrix = to_matrix(d_array, 3, 2, 1);
  transformed_param_matrix = to_matrix(d_iarray, 3, 2);
  transformed_param_matrix = to_matrix(d_iarray, 3, 2, 1);

  transformed_param_matrix = to_matrix(d_array2);
  transformed_param_matrix = to_matrix(d_iarray2);


  transformed_param_matrix = to_matrix(p_matrix);
  transformed_param_matrix = to_matrix(p_vector);
  transformed_param_matrix = to_matrix(p_row_vector);

  transformed_param_matrix = to_matrix(p_matrix, 4, 2);
  transformed_param_matrix = to_matrix(p_vector, 3, 5);
  transformed_param_matrix = to_matrix(p_row_vector, 2, 4);

  transformed_param_matrix = to_matrix(p_matrix, 4, 2, 1);
  transformed_param_matrix = to_matrix(p_vector, 3, 5, 1);
  transformed_param_matrix = to_matrix(p_row_vector, 2, 4, 1);

  transformed_param_matrix = to_matrix(p_array, 3, 2);
  transformed_param_matrix = to_matrix(p_array, 3, 2, 1);

  transformed_param_matrix = to_matrix(p_array2);
}
model {
  y_p ~ normal(0,1);
}
