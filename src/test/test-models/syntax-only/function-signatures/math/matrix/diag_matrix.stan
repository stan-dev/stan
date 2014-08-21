data { 
  int d_int;
  vector[d_int] d_vector;
}

transformed data {
  matrix[d_int,d_int] transformed_data_matrix;

  transformed_data_matrix <- diag_matrix(d_vector);
}
parameters {
  real y_p;
  vector[d_int] p_vector;
}
transformed parameters {
  matrix[d_int,d_int] transformed_param_matrix;

  transformed_param_matrix <- diag_matrix(d_vector);
  transformed_param_matrix <- diag_matrix(p_vector);
}
model {  
  y_p ~ normal(0,1);
}
