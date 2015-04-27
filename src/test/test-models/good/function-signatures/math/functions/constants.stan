data { 
  int d_int;
  real d_real;
}
transformed data {
  real transformed_data_real;

  transformed_data_real <- pi();
  transformed_data_real <- e();
  transformed_data_real <- sqrt2();
  transformed_data_real <- log2();
  transformed_data_real <- log10();
}
parameters {
  real p_real;
  real y_p;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real <- pi();
  transformed_param_real <- e();
  transformed_param_real <- sqrt2();
  transformed_param_real <- log2();
  transformed_param_real <- log10();
}
model {  
  y_p ~ normal(0,1);
}
