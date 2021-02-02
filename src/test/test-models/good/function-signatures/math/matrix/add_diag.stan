functions {

}
data {
  int<lower=1> N;
  matrix[N, N] K;
  row_vector[N] row_vec;
  vector[N] vec;
  int scalar;
}
transformed data {
  matrix[N, N] K1;
  matrix[N, N] K2;
  matrix[N, N] K3;
  K1 = add_diag(K, row_vec);
  K2 = add_diag(K, vec);
  K3 = add_diag(K, scalar);
}
parameters {
  matrix[N, N] Kp;
  row_vector[N] row_vec_p;
  vector[N] vec_p;
  real scalar_p;
}
model {
  matrix[N, N] K4 = add_diag(Kp, row_vec_p);
  matrix[N, N] K5 = add_diag(Kp, vec_p);
  matrix[N, N] K6 = add_diag(Kp, scalar_p);
  matrix[N, N] K7 = add_diag(K, row_vec_p);
  matrix[N, N] K8 = add_diag(K, vec_p);
  matrix[N, N] K9 = add_diag(K, scalar_p);
  matrix[N, N] K10 = add_diag(Kp, row_vec);
  matrix[N, N] K11 = add_diag(Kp, vec);
  matrix[N, N] K12 = add_diag(Kp, scalar);
}
generated quantities {

}

