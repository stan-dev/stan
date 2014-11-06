data { 
  int d_int;
  int d_int_array[d_int];
  real d_real_array[d_int];
  matrix[d_int,d_int] d_matrix;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}

transformed data {
  matrix[d_int,d_int] transformed_data_matrix;
  vector[d_int] transformed_data_vector;
  row_vector[d_int] transformed_data_row_vector;

  transformed_data_matrix <- log(d_matrix);
  transformed_data_vector <- log(d_vector);
  transformed_data_row_vector <- log(d_row_vector);
}
parameters {
  real p_real;
  real y_p;
  real p_real_array[d_int];
  matrix[d_int,d_int] p_matrix;
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;
}
transformed parameters {
  matrix[d_int,d_int] transformed_param_matrix;
  vector[d_int] transformed_param_vector;
  row_vector[d_int] transformed_param_row_vector;

  transformed_param_matrix <- log(d_matrix);
  transformed_param_vector <- log(d_vector);
  transformed_param_row_vector <- log(d_row_vector);
  transformed_param_matrix <- log(p_matrix);
  transformed_param_vector <- log(p_vector);
  transformed_param_row_vector <- log(p_row_vector);
}
model {  
  y_p ~ normal(0,1);
}
