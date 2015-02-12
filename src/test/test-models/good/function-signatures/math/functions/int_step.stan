data { 
  int d_int;
  matrix[d_int,d_int] d_matrix;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}
transformed data {
  int transformed_data_int;

  transformed_data_int <- int_step(d_int);
}
parameters {
  matrix[d_int,d_int] p_matrix;
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;
  real y_p;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real <- int_step(rows(d_vector)); // using int_step to test integer output
  transformed_param_real <- int_step(rows(p_vector)); 
  transformed_param_real <- int_step(rows(d_row_vector));
  transformed_param_real <- int_step(rows(p_row_vector));
  transformed_param_real <- int_step(rows(d_matrix));
  transformed_param_real <- int_step(rows(p_matrix));
  transformed_param_real <- int_step(cols(d_vector)); // using int_step to test integer output
  transformed_param_real <- int_step(cols(p_vector)); 
  transformed_param_real <- int_step(cols(d_row_vector));
  transformed_param_real <- int_step(cols(p_row_vector));
  transformed_param_real <- int_step(cols(d_matrix));
  transformed_param_real <- int_step(cols(p_matrix));
}
model {  
  y_p ~ normal(0,1);
}
