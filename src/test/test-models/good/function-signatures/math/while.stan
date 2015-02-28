data { 
  int d_int;
}
transformed data {
  int transformed_data_int;
  real transformed_data_real;
  real transformed_data_real2;

  while (transformed_data_real) transformed_data_real2 <- 1.0;

  while (transformed_data_real) {
    transformed_data_real2 <- 1.0;
    transformed_data_real2 <- 2.0;
  }

  while (transformed_data_real)
    transformed_data_real2 <- 1.0;
}
parameters {
  real y_p;
}
transformed parameters {
  real transformed_param_real;

  while (transformed_param_real) transformed_param_real <- 1.0;

  while (transformed_data_real) {
    transformed_param_real <- 1.0;
    transformed_param_real <- 2.0;
  }

  while (transformed_param_real) {
    transformed_param_real <- 1.0;
    transformed_param_real <- 2.0;
  }

  while (transformed_data_real)
    transformed_param_real <- 1.0;

  while (transformed_param_real)
    transformed_param_real <- 1.0;

}
model {  
  y_p ~ normal(0,1);
}
