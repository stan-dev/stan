data { 
  int d_int;
  matrix[d_int,d_int] d_matrix;
}

transformed data {
  int transformed_data_array[d_int];
  transformed_data_array <- csr_extract_u(d_matrix);
}

parameters {
  real y_p;
}

model {  
  y_p ~ normal(0,1);
}
