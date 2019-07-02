functions {}
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
model {}
generated quantities {}
