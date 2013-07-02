data {
  int r;
  int T;
  matrix[r, T] y;
  vector[r] m0;
  cov_matrix[r] C0;
}
transformed data {
  vector[r] ones;
  matrix[r, r] G;
  matrix[r, r] F;
  for (i in 1:r) {
    ones[i] <- 1.0;
  }
  G <- diag_matrix(ones);
  F <- G;
}
parameters {
  real<lower=-1.0, upper=1.0> rho;
  vector<lower=0.0>[r] sigma;
  vector<lower=0.0>[r] W_diag;
}
transformed parameters {
  cov_matrix[r] V;
  cov_matrix[r] W;
  W <- diag_matrix(W_diag);
  for (i in 1:r) {
    V[i, i] <- pow(sigma[i], 2);
    for (j in 1:(i - 1)) {
      V[i, j] <- sigma[i] * sigma[j] * rho;
      V[j, i] <- V[i, j];
    }
  }
}
model {
  y ~ gaussian_dlm_obs(F, G, V, W, m0, C0);
}
