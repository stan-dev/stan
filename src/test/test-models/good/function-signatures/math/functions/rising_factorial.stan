data { 
  int d_int;
  int r_int;
  real d_real;
}
transformed data {
  real transformed_data_real;

  transformed_data_real <- rising_factorial(d_real, d_int);
  transformed_data_real <- rising_factorial(r_int, d_int);
}
parameters {
  real p_real;
  real y_p;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real <-  rising_factorial(d_real, d_int);
  transformed_param_real <-  rising_factorial(r_int, d_int);
  transformed_param_real <-  rising_factorial(p_real,r_int);
}
model {  
  y_p ~ normal(0,1);
}
