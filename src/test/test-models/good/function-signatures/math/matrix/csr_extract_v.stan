data { 
  int d_int;
  matrix[d_int,d_int] d_matrix;
}

transformed data {
  vector[d_int] transformed_data_vector;
  transformed_data_vector <- csr_extract_v(d_matrix);
}

parameters {
  real y_p;
}
transformed parameters {
  transformed_data_vector <- csr_extract_v(p_matrix);
}
model {  
  y_p ~ normal(0,1);
}
