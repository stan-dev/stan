
data { 
  int d_int;
  matrix[d_int,d_int] d_matrix;
}

transformed data {
  vector[d_int] transformed_data_vector_w;
  int transformed_data_array_v[d_int];
  int transformed_data_array_u[d_int];
  int transformed_data_array_z[d_int];

  vector[d_int] transformed_d_vector;
  vector[d_int] transformed_d_vector_2;

  transformed_d_vector <- csr_matrix_times_vector(
    d_int, d_int,
    transformed_data_vector_w,
    transformed_data_array_v,
    transformed_data_array_u,
    transformed_data_array_z,
    transformed_d_vector_2
  );
}

parameters {
  real y_p;

  vector[d_int] p_vector;
}

transformed parameters {
  matrix[d_int, d_int] transformed_p_matrix;
  vector[d_int] transformed_p_vector;
  vector[d_int] transformed_param_vector_w;

  transformed_p_vector <- csr_matrix_times_vector(
    d_int, d_int,
    transformed_param_vector_w,
    transformed_data_array_v,
    transformed_data_array_u,
    transformed_data_array_z,
    transformed_d_vector
  );

  transformed_p_vector <- csr_matrix_times_vector(
    d_int, d_int,
    transformed_data_vector_w,
    transformed_data_array_v,
    transformed_data_array_u,
    transformed_data_array_z,
    p_vector
  );
}

model {  
  y_p ~ normal(0,1);
}
