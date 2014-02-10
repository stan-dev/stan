transformed data {
  real rho;
  matrix[3, 3] Omega;

  rho <- 0.9;

  Omega[1, 1] <- 1;
  Omega[2, 2] <- 1;
  Omega[3, 3] <- 1;
  
  Omega[1, 2] <- rho;
  Omega[2, 1] <- rho;
  Omega[2, 3] <- rho;
  Omega[3, 2] <- rho;
  
  Omega[1, 3] <- rho * rho;
  Omega[3, 1] <- rho * rho;

}

parameters {
  vector[3] x;
}

model {
  x ~ multi_normal(rep_vector(0, 3), Omega);
}

generated quantities {
  real x2[3];
  real xy[3];

  x2[1] <- x[1] * x[1];
  x2[2] <- x[2] * x[2];
  x2[3] <- x[3] * x[3];
  
  xy[1] <- x[1] * x[2];
  xy[2] <- x[2] * x[3];
  xy[3] <- x[1] * x[3];
}