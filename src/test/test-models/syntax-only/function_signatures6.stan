data { 
  int d_int;
  int d_int_array[d_int];
  real d_real;
  real d_real_array[d_int];
  matrix[d_int,d_int] d_matrix;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}
transformed data {
  int transformed_data_int;
  real transformed_data_real;
  real transformed_data_real_array[d_int];
  matrix[d_int,d_int] transformed_data_matrix;
  vector[d_int] transformed_data_vector;
  row_vector[d_int] transformed_data_row_vector;

  // Vector Probabilities
  // Covariance Matrix Distributions
  // mdivide_..._low
  transformed_data_matrix <- mdivide_left_tri_low(d_matrix,d_matrix);
  transformed_data_vector <- mdivide_left_tri_low(d_matrix,d_vector);
  transformed_data_matrix <- mdivide_right_tri_low(d_matrix,d_matrix);
  transformed_data_row_vector <- mdivide_right_tri_low(d_row_vector,d_matrix);
}
parameters {
  real p_real;
  real p_real_array[d_int];
  matrix[d_int,d_int] p_matrix;
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;
}
transformed parameters {
  real transformed_param_real;
  real transformed_param_real_array[d_int];
  matrix[d_int,d_int] transformed_param_matrix;
  vector[d_int] transformed_param_vector;
  row_vector[d_int] transformed_param_row_vector;

  // Vector Probabilities

  // mdivide_..._low
  transformed_param_matrix <- mdivide_left_tri_low(d_matrix,d_matrix);
  transformed_param_matrix <- mdivide_left_tri_low(transformed_param_matrix,d_matrix);
  transformed_param_matrix <- mdivide_left_tri_low(transformed_param_matrix,
                                                   transformed_param_matrix);
  transformed_param_matrix <- mdivide_left_tri_low(d_matrix,
                                                   transformed_param_matrix);

  transformed_param_vector <- mdivide_left_tri_low(d_matrix,d_vector);
  transformed_param_vector <- mdivide_left_tri_low(transformed_param_matrix,d_vector);
  transformed_param_vector <- mdivide_left_tri_low(transformed_param_matrix,
                                                   transformed_param_vector);
  transformed_param_vector <- mdivide_left_tri_low(d_matrix,
                                                   transformed_param_vector);

  transformed_param_matrix <- mdivide_right_tri_low(d_matrix,d_matrix);
  transformed_param_matrix <- mdivide_right_tri_low(transformed_param_matrix,d_matrix);
  transformed_param_matrix <- mdivide_right_tri_low(transformed_param_matrix,
                                                    transformed_param_matrix);
  transformed_param_matrix <- mdivide_right_tri_low(d_matrix,
                                                    transformed_param_matrix);

  transformed_param_row_vector <- mdivide_right_tri_low(d_row_vector,d_matrix);
  transformed_param_row_vector <- mdivide_right_tri_low(transformed_param_row_vector,d_matrix);
  transformed_param_row_vector <- mdivide_right_tri_low(transformed_param_row_vector,
                                                        transformed_param_matrix);
  transformed_param_row_vector <- mdivide_right_tri_low(d_row_vector,
                                                        transformed_param_matrix);

}
model {  
}
