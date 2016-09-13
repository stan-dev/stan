data { 
  int d_int;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
  real d_real;
}
transformed data {
  real transformed_data_real;

  transformed_data_real <- squared_distance(d_vector, d_vector);
  transformed_data_real <- squared_distance(d_vector, d_row_vector);
  transformed_data_real <- squared_distance(d_row_vector, d_vector);
  transformed_data_real <- squared_distance(d_row_vector, d_row_vector);
  transformed_data_real <- squared_distance(d_real, d_real);
}
parameters {
  real y_p;
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;
  real p_real;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real <- squared_distance(d_vector, d_vector);
  transformed_param_real <- squared_distance(d_vector, d_row_vector);
  transformed_param_real <- squared_distance(d_row_vector, d_vector);
  transformed_param_real <- squared_distance(d_row_vector, d_row_vector);

  transformed_param_real <- squared_distance(p_vector, d_vector);
  transformed_param_real <- squared_distance(p_vector, d_row_vector);
  transformed_param_real <- squared_distance(p_row_vector, d_vector);
  transformed_param_real <- squared_distance(p_row_vector, d_row_vector);

  transformed_param_real <- squared_distance(d_vector, p_vector);
  transformed_param_real <- squared_distance(d_vector, p_row_vector);
  transformed_param_real <- squared_distance(d_row_vector, p_vector);
  transformed_param_real <- squared_distance(d_row_vector, p_row_vector);

  transformed_param_real <- squared_distance(p_vector, p_vector);
  transformed_param_real <- squared_distance(p_vector, p_row_vector);
  transformed_param_real <- squared_distance(p_row_vector, p_vector);
  transformed_param_real <- squared_distance(p_row_vector, p_row_vector);

  transformed_param_real <- squared_distance(p_real, p_real);
}
model {  
  y_p ~ normal(0, 1);
}
