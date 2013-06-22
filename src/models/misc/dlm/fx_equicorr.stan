data {
  int n;
  int T;
  matrix[n, T] y;
}
transformed data {
  vector[n] ones;
  matrix[n, n] G;
  matrix[n, n] F;
  for (i in 1:n) {
    ones[i] <- 1.0;
  }
  G <- diag_matrix(ones);
  F <- G;
}
parameters {
  real<lower=-1.0, upper=1.0> rho;
  vector<lower=0.0>[n] sigma;
  vector<lower=0.0>[n] W_diag;
}
transformed parameters {
  cov_matrix[n] V;
  cov_matrix[n] W;
  W <- diag_matrix(W_diag);
  for (i in 1:n) {
    V[i, i] <- pow(sigma[i], 2);
    for (j in 1:(i - 1)) {
      V[i, j] <- sigma[i] * sigma[j] * rho;
      V[j, i] <- V[i, j];
    }
  }
}
model {
  y ~ gaussian_dlm_log(F, G, V, W);
}
