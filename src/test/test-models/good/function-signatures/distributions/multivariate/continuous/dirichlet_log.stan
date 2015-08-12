data { 
  int d_int;
  vector[d_int] d_vector;
  row_vector[d_int] d_row_vector;
}
transformed data {
  real transformed_data_real;

  transformed_data_real <- dirichlet_log(d_vector, d_vector);
}
parameters {
  vector[d_int] p_vector;
  row_vector[d_int] p_row_vector;

  real y_p;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real <- dirichlet_log(d_vector, d_vector);
  transformed_param_real <- dirichlet_log(d_vector, p_vector);
  transformed_param_real <- dirichlet_log(p_vector, d_vector);
  transformed_param_real <- dirichlet_log(p_vector, p_vector);
}
model {  
  y_p ~ normal(0,1);
}
