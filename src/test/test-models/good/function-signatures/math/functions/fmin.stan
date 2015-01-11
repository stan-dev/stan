data { 
  int d_int;
  int r_int;
  real d_real;
  real r_real;
  
}
transformed data {
  int transformed_data_int;
  real transformed_data_real;

  transformed_data_real <- fmin(d_real, r_real);
  transformed_data_real <- fmin(d_int, r_real);
  transformed_data_real <- fmin(d_real, d_int);
  transformed_data_real <- fmin(r_int, d_int);
}
parameters {
  real p_real;
  real y_p;
}
transformed parameters {
  real transformed_param_real;

  transformed_param_real <-  fmin(d_real, r_real);
  transformed_param_real <-  fmin(d_int , r_real);
  transformed_param_real <-  fmin(d_real, d_int);
  transformed_param_real <-  fmin(r_int, d_int);

  transformed_param_real <-  fmin(r_int, p_real);
  transformed_param_real <-  fmin(r_real,p_real);
  transformed_param_real <-  fmin(p_real,p_real);
  transformed_param_real <-  fmin(p_real,r_int);
  transformed_param_real <-  fmin(p_real,r_real);
}
model {  
  y_p ~ normal(0,1);
}
