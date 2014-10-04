data { 
  int d_int;
  real d_real;
}
transformed data {
  real transformed_data_real;

  transformed_data_real <- if_else(is_nan(d_real), d_int, d_int);
  transformed_data_real <- if_else(is_nan(d_real), d_int, d_real);
  transformed_data_real <- if_else(is_nan(d_real), d_real, d_real);
  transformed_data_real <- if_else(is_nan(d_real), d_real, d_int);
}
parameters {
  real p_real;
  real y_p;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real <-  if_else(is_nan(d_real), d_int, d_int);;
  transformed_param_real <-  if_else(is_nan(d_real), d_int, d_real);
  transformed_param_real <-  if_else(is_nan(d_real), d_real, d_real);
  transformed_param_real <-  if_else(is_nan(d_real), d_real, d_int);

  transformed_param_real <-  if_else(is_nan(d_real), d_int, p_real);
  transformed_param_real <-  if_else(is_nan(d_real), p_real, d_int);
  transformed_param_real <-  if_else(is_nan(d_real), d_real, p_real);
  transformed_param_real <-  if_else(is_nan(d_real), p_real, d_real);
  transformed_param_real <-  if_else(is_nan(d_real), p_real, p_real);
}
model {  
  y_p ~ normal(0,1);
}
