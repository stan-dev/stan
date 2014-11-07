data { 
  int d_int;
  real d_real;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}

transformed data {
  matrix[d_int,d_int] transformed_data_matrix;

  transformed_data_matrix <- rep_matrix(d_real, d_int, d_int);
  transformed_data_matrix <- rep_matrix(d_vector, d_int);
  transformed_data_matrix <- rep_matrix(d_row_vector, d_int);
}
parameters {
  real p_real;
  real y_p;
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;
}
transformed parameters {
  matrix[d_int,d_int] transformed_param_matrix;

  transformed_param_matrix <- rep_matrix(d_real, d_int, d_int);
  transformed_param_matrix <- rep_matrix(d_vector, d_int);
  transformed_param_matrix <- rep_matrix(d_row_vector, d_int);
  transformed_param_matrix <- rep_matrix(p_real, d_int, d_int);
  transformed_param_matrix <- rep_matrix(p_vector, d_int);
  transformed_param_matrix <- rep_matrix(p_row_vector, d_int);
}
model {  
  y_p ~ normal(0,1);
}
