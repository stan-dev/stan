transformed data {
  matrix[3, 4] a_td;
  vector[5] w_d;
  int v[6];
  int u[7];
  
  a_td <- csr_to_dense_matrix(3, 4, w_d, v, u);  
}
parameters {
  real y_p;
  vector[5] w_p;
}
transformed parameters {
  matrix[3,4] a_tp;
  a_tp <- csr_to_dense_matrix(3, 4, w_d, v, u);
  a_tp <- csr_to_dense_matrix(3, 4, w_p, v, u);
}
model {  
  y_p ~ normal(0, 1);
}
