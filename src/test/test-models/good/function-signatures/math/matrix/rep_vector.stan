data { 
  int d_int;
  real d_real;
}

transformed data {
  vector[d_int] transformed_data_vector;

  transformed_data_vector <- rep_vector(d_real, d_int);
}
parameters {
  real p_real;
  real y_p;
}
transformed parameters {
  vector[d_int] transformed_param_vector;

  transformed_param_vector <- rep_vector(d_real, d_int);
  transformed_param_vector <- rep_vector(p_real, d_int);
}
model {  
  y_p ~ normal(0,1);
}
