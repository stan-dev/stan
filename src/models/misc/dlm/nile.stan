data {
  int n;
  matrix[1, n] y;
}
transformed data {
  matrix[1, 1] F;
  matrix[1, 1] G;
  F[1, 1] <- 1;
  G[1, 1] <- 1;
}
parameters {
  vector<lower=0.0>[1] V;
  cov_matrix[1] W;
}
model {
  y ~ gaussian_dlm_log(F, G, V, W);
}
