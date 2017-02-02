data { 
  int d_int;
  matrix[d_int,d_int] d_matrix;
  real d_array[6];
}

transformed data {
  matrix[d_int,d_int] transformed_data_matrix;

  transformed_data_matrix = to_matrix(d_matrix);
  transformed_data_matrix = to_matrix(d_array, 2, 3);
}
parameters {
  real y_p;
  matrix[d_int,d_int] p_matrix;
  real p_array[6];
}
transformed parameters {
  matrix[d_int,d_int] transformed_param_matrix;

  transformed_param_matrix = to_matrix(d_matrix);
  transformed_param_matrix = to_matrix(p_matrix);
  transformed_param_matrix = to_matrix(d_array, 3, 2);
  transformed_param_matrix = to_matrix(p_array, 3, 2);
}
model {  
  y_p ~ normal(0,1);
}
