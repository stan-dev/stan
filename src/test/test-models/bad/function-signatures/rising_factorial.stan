data { 
  real d_real;
  real r_real;
}
transformed parameters {
  real transformed_param_real;
  transformed_param_real <-  rising_factorial(d_real, r_real);
}
