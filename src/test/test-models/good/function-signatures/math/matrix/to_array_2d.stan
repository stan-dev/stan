data {
  int d_int;
  matrix[d_int, d_int] d_matrix;
}
transformed data {
  array[d_int, d_int] real transformed_data_real_array_2;
  transformed_data_real_array_2 = to_array_2d(d_matrix);
}
parameters {
  real y_p;
  matrix[d_int, d_int] p_matrix;
}
transformed parameters {
  array[d_int, d_int] real transformed_param_real_array_2;
  transformed_param_real_array_2 = to_array_2d(d_matrix);
  transformed_param_real_array_2 = to_array_2d(p_matrix);
}
model {
  y_p ~ normal(0, 1);
}

