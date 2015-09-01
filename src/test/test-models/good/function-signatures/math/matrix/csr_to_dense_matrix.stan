//add("csr_to_dense_matrix", MATRIX_T,INT_T,INT_T,
//        VECTOR_T,int_vector_types[1],int_vector_types[1],int_vector_types[1]);


data { 
  int d_int;
  matrix[d_int,d_int] d_matrix;
}

transformed data {
  vector[d_int] transformed_data_vector;
  transformed_data_vector <- csr_extract_w(d_matrix);

  csr_to_dense_matrix(transformed_data_vector
}

parameters {
  real y_p;
  matrix[d_int,d_int] p_matrix;
}
transformed parameters {
  vector[d_int] transformed_param_vector;

  transformed_param_vector <- csr_extract_w(d_matrix);
  transformed_param_vector <- csr_extract_w(p_matrix);
}
model {  
  y_p ~ normal(0,1);
}
