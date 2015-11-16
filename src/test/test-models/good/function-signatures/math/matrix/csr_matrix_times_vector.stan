data { 
  int m;
  int n;
  int v[3];
  int u[3];
  vector[3] w_d;
  vector[3] b_d;
}
transformed data {
  vector[3] ab_d;
  ab_d <- csr_matrix_times_vector(m, n, w_d, v, u, b_d);
}
parameters {
  real y_p;
  vector[3] w_p;
  vector[3] b_p;
}
transformed parameters {
  vector[3] ab_p;
  ab_p <- csr_matrix_times_vector(m, n, w_d, v, u, b_d);
  ab_p <- csr_matrix_times_vector(m, n, w_d, v, u, b_p);
  ab_p <- csr_matrix_times_vector(m, n, w_p, v, u, b_d);
  ab_p <- csr_matrix_times_vector(m, n, w_p, v, u, b_p);
}
model {  
  y_p ~ normal(0,1);
}
