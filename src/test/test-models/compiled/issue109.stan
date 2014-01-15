parameters {
  real y;
}
model {
  y ~ normal(0, 1);
}

generated quantities {
  real z[2,2];
  matrix[2,2] z_mat;

  z[1,1] <- 1;
  z[1,2] <- 2;
  z[2,1] <- 3;
  z[2,2] <- 4;

  z_mat[1,1] <- 1;
  z_mat[1,2] <- 2;
  z_mat[2,1] <- 3;
  z_mat[2,2] <- 4;
}
