data { 
  int d_int;
  matrix[d_int,d_int] d_matrix;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}

transformed data {
  int transformed_data_int;
  real transformed_data_real;

  transformed_data_int <- cols(d_vector);
  transformed_data_int <- cols(d_row_vector);
  transformed_data_int <- cols(d_matrix);
  transformed_data_real <- cols(d_vector);
  transformed_data_real <- cols(d_row_vector);
  transformed_data_real <- cols(d_matrix);
}
parameters {
  real y_p;
  matrix[d_int,d_int] p_matrix;
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real <- cols(d_vector);
  transformed_param_real <- cols(d_row_vector);
  transformed_param_real <- cols(d_matrix);
  transformed_param_real <- cols(p_vector);
  transformed_param_real <- cols(p_row_vector);
  transformed_param_real <- cols(p_matrix);
}
model {  
  y_p ~ normal(0,1);
}
