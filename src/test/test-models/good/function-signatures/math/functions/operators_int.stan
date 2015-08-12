data { 
  int d_int;
}
transformed data {
  int transformed_data_int;

  transformed_data_int <- d_int + d_int;
  transformed_data_int <- d_int - d_int;
  transformed_data_int <- d_int * d_int;
  transformed_data_int <- d_int / d_int;

  transformed_data_int <- -d_int;
  transformed_data_int <- +d_int;
}
parameters {
  real y_p;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real <- d_int + d_int;
  transformed_param_real <- d_int - d_int;
  transformed_param_real <- d_int * d_int;
  transformed_param_real <- d_int / d_int;

  transformed_param_real <- -d_int;
  transformed_param_real <- +d_int;
}
model {  
  y_p ~ normal(0,1);
}
