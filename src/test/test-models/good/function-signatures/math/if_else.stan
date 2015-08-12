data { 
  int d_int;
}
transformed data {
  int transformed_data_int;
  real transformed_data_real;
  real transformed_data_real2;


  if (transformed_data_real)
    transformed_data_real2 <- 1.0;
  else if (transformed_data_real)
    transformed_data_real2 <- 2.0;
  else if (transformed_data_real)
    transformed_data_real2 <- 3.0;

  if (transformed_data_real)
    transformed_data_real <- 1.0;
  else
    transformed_data_real <- 2.0;

  if (transformed_data_real) {
    transformed_data_real <- 1.0;
    transformed_data_real <- 1.0;
  } else if (transformed_data_real) {
    transformed_data_real <- 1.0;
    transformed_data_real <- 1.0;
  } else if (transformed_data_real) {
    transformed_data_real <- 1.0;
    transformed_data_real <- 1.0;
  } else {
    transformed_data_real <- 1.0;
    transformed_data_real <- 1.0;
  }    

  if (transformed_data_int) {
    transformed_data_real <- 1.0;
    transformed_data_real <- 1.0;
  } else if (transformed_data_int) {
    transformed_data_real <- 1.0;
    transformed_data_real <- 1.0;
  } else if (transformed_data_int) {
    transformed_data_real <- 1.0;
    transformed_data_real <- 1.0;
  } else {
    transformed_data_real <- 1.0;
    transformed_data_real <- 1.0;
  }    
}
parameters {
  real y_p;
}
transformed parameters {
  real transformed_param_real;


  if (transformed_param_real)
    transformed_param_real <- 1.0;
  else if (transformed_param_real)
    transformed_param_real <- 2.0;
  else if (transformed_param_real)
    transformed_param_real <- 3.0;

  if (transformed_param_real)
    transformed_param_real <- 1.0;
  else
    transformed_param_real <- 2.0;

  if (transformed_param_real) {
    transformed_param_real <- 1.0;
    transformed_param_real <- 1.0;
  } else if (transformed_param_real) {
    transformed_param_real <- 1.0;
    transformed_param_real <- 1.0;
  } else if (transformed_param_real) {
    transformed_param_real <- 1.0;
    transformed_param_real <- 1.0;
  } else {
    transformed_param_real <- 1.0;
    transformed_param_real <- 1.0;
  }    

  if (transformed_data_int) {
    transformed_param_real <- 1.0;
    transformed_param_real <- 1.0;
  } else if (transformed_data_int) {
    transformed_param_real <- 1.0;
    transformed_param_real <- 1.0;
  } else if (transformed_data_int) {
    transformed_param_real <- 1.0;
    transformed_param_real <- 1.0;
  } else {
    transformed_param_real <- 1.0;
    transformed_param_real <- 1.0;
  }    

}
model {  
  y_p ~ normal(0,1);
}
