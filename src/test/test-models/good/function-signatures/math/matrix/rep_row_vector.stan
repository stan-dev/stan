data { 
  int d_int;
  real d_real;
}

transformed data {
  row_vector[d_int] transformed_data_row_vector;

  transformed_data_row_vector <- rep_row_vector(d_real, d_int);
}
parameters {
  real p_real;
  real y_p;
}
transformed parameters {
  row_vector[d_int] transformed_param_row_vector;

  transformed_param_row_vector <- rep_row_vector(d_real, d_int);
  transformed_param_row_vector <- rep_row_vector(p_real, d_int);
}
model {  
  y_p ~ normal(0,1);
}
