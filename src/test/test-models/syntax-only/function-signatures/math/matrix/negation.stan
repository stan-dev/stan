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

  transformed_data_vector <- -d_vector;
  transformed_data_row_vector <- -d_row_vector;
  transformed_data_matrix <- -d_matrix;
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

  transformed_param_vector <- -d_vector;
  transformed_param_row_vector <- -d_row_vector;
  transformed_param_matrix <- -d_matrix;
  transformed_param_vector <- -p_vector;
  transformed_param_row_vector <- -p_row_vector;
  transformed_param_matrix <- -p_matrix;
}
model {  
  y_p ~ normal(0,1);
}
