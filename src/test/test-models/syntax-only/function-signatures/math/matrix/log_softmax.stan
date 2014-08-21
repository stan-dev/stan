data { 
  int d_int;
  vector[d_int] d_vector;
}

transformed data {
  vector[d_int] transformed_data_vector;

  transformed_data_vector <- log_softmax(d_vector);
}
parameters {
  real y_p;
  vector[d_int] p_vector;
}
transformed parameters {
  vector[d_int] transformed_param_vector;

  transformed_param_vector <- log_softmax(d_vector);
  transformed_param_vector <- log_softmax(p_vector);
}
model {  
  y_p ~ normal(0,1);
}
