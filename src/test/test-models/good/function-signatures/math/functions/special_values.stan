data { 
  int d_int;
  real d_real;
}
transformed data {
  real transformed_data_real;

  transformed_data_real <- not_a_number();
  transformed_data_real <- positive_infinity();
  transformed_data_real <- negative_infinity();
  transformed_data_real <- machine_precision();
}
parameters {
  real p_real;
  real y_p;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real <- not_a_number();
  transformed_param_real <- positive_infinity();
  transformed_param_real <- negative_infinity();
  transformed_param_real <- machine_precision();
}
model {  
  y_p ~ normal(0,1);
}
