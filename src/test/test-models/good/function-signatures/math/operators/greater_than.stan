data { 
  int d_int;
}
transformed data {
  int transformed_data_int;
  real transformed_data_real;

  transformed_data_int <- transformed_data_int > transformed_data_int;
  transformed_data_int <- transformed_data_int > transformed_data_real;
  transformed_data_int <- transformed_data_real > transformed_data_int;
  transformed_data_int <- transformed_data_real > transformed_data_real;

  transformed_data_real <- transformed_data_int > transformed_data_int;
  transformed_data_real <- transformed_data_int > transformed_data_real;
  transformed_data_real <- transformed_data_real > transformed_data_int;
  transformed_data_real <- transformed_data_real > transformed_data_real;
}
parameters {
  real y_p;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real <- transformed_data_int > transformed_data_int;
  transformed_param_real <- transformed_data_int > transformed_data_real;
  transformed_param_real <- transformed_data_real > transformed_data_int;
  transformed_param_real <- transformed_data_real > transformed_data_real;

  transformed_param_real <- transformed_data_int > transformed_param_real;
  transformed_param_real <- transformed_param_real > transformed_data_int;
  transformed_param_real <- transformed_param_real > transformed_data_real;
  transformed_param_real <- transformed_param_real > transformed_param_real;
  transformed_param_real <- transformed_data_real > transformed_param_real;
}
model {  
  y_p ~ normal(0,1);
}
