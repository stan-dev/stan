data { 
  int d_int;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}
transformed data {
  real transformed_data_real;
  vector[d_int] transformed_data_vector;
  row_vector[d_int] transformed_data_row_vector;

  transformed_data_real <- distance(d_vector, d_vector);
  transformed_data_real <- distance(d_vector, d_row_vector);
  transformed_data_real <- distance(d_row_vector, d_vector);
  transformed_data_real <- distance(d_row_vector, d_row_vector);
}
parameters {
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;
}
transformed parameters {
  real transformed_param_real;
  vector[d_int] transformed_param_vector;
  row_vector[d_int] transformed_param_row_vector;

  transformed_param_real <- distance(transformed_param_vector, 
                                     transformed_param_vector);
  transformed_param_real <- distance(transformed_param_vector, 
                                     transformed_param_row_vector);
  transformed_param_real <- distance(transformed_param_row_vector, 
                                     transformed_param_vector);
  transformed_param_real <- distance(transformed_param_row_vector, 
                                     transformed_param_row_vector);
}
model {  
}
