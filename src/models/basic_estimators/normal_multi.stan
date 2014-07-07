data {
  int<lower=0> N; //response variable size
  int<lower=0> J; //number of observations
  vector[N] y[J];
  vector[N] z[J];
  matrix[N, N] sigma;
}
transformed data {
  row_vector[N] ry[J];
  row_vector[N] rz[J];
  matrix[N, N] inv_sigma;
  for (n in 1:N)
    for (j in 1:J) {
      ry[j][n] <- y[j][n];
      rz[j][n] <- z[j][n];
    }
  inv_sigma <- inverse_spd(sigma);
}
parameters {
  vector[N] beta[J];
  row_vector[N] rbeta[J];
}
model {
  y ~ multi_normal(beta, sigma);
  y ~ multi_normal(beta[1], sigma);
  y[1] ~ multi_normal(beta, sigma);
  y[1] ~ multi_normal(beta[1], sigma);
  y ~ multi_normal(z, sigma);
  y ~ multi_normal(z[1], sigma);
  y[1] ~ multi_normal(z, sigma);
  y[1] ~ multi_normal(z[1], sigma);
  
  y ~ multi_normal(rbeta, sigma);
  y ~ multi_normal(rbeta[1], sigma);
  y[1] ~ multi_normal(rbeta, sigma);
  y[1] ~ multi_normal(rbeta[1], sigma);
  y ~ multi_normal(rz, sigma);
  y ~ multi_normal(rz[1], sigma);
  y[1] ~ multi_normal(rz, sigma);
  y[1] ~ multi_normal(rz[1], sigma);
  
  ry ~ multi_normal(beta, sigma);
  ry ~ multi_normal(beta[1], sigma);
  ry[1] ~ multi_normal(beta, sigma);
  ry[1] ~ multi_normal(beta[1], sigma);
  ry ~ multi_normal(z, sigma);
  ry ~ multi_normal(z[1], sigma);
  ry[1] ~ multi_normal(z, sigma);
  ry[1] ~ multi_normal(z[1], sigma);
  
  ry ~ multi_normal(rbeta, sigma);
  ry ~ multi_normal(rbeta[1], sigma);
  ry[1] ~ multi_normal(rbeta, sigma);
  ry[1] ~ multi_normal(rbeta[1], sigma);
  ry ~ multi_normal(rz, sigma);
  ry ~ multi_normal(rz[1], sigma);
  ry[1] ~ multi_normal(rz, sigma);
  ry[1] ~ multi_normal(rz[1], sigma);



  y ~ multi_normal_cholesky(beta, sigma);
  y ~ multi_normal_cholesky(beta[1], sigma);
  y[1] ~ multi_normal_cholesky(beta, sigma);
  y[1] ~ multi_normal_cholesky(beta[1], sigma);
  y ~ multi_normal_cholesky(z, sigma);
  y ~ multi_normal_cholesky(z[1], sigma);
  y[1] ~ multi_normal_cholesky(z, sigma);
  y[1] ~ multi_normal_cholesky(z[1], sigma);
  
  y ~ multi_normal_cholesky(rbeta, sigma);
  y ~ multi_normal_cholesky(rbeta[1], sigma);
  y[1] ~ multi_normal_cholesky(rbeta, sigma);
  y[1] ~ multi_normal_cholesky(rbeta[1], sigma);
  y ~ multi_normal_cholesky(rz, sigma);
  y ~ multi_normal_cholesky(rz[1], sigma);
  y[1] ~ multi_normal_cholesky(rz, sigma);
  y[1] ~ multi_normal_cholesky(rz[1], sigma);
  
  ry ~ multi_normal_cholesky(beta, sigma);
  ry ~ multi_normal_cholesky(beta[1], sigma);
  ry[1] ~ multi_normal_cholesky(beta, sigma);
  ry[1] ~ multi_normal_cholesky(beta[1], sigma);
  ry ~ multi_normal_cholesky(z, sigma);
  ry ~ multi_normal_cholesky(z[1], sigma);
  ry[1] ~ multi_normal_cholesky(z, sigma);
  ry[1] ~ multi_normal_cholesky(z[1], sigma);
  
  ry ~ multi_normal_cholesky(rbeta, sigma);
  ry ~ multi_normal_cholesky(rbeta[1], sigma);
  ry[1] ~ multi_normal_cholesky(rbeta, sigma);
  ry[1] ~ multi_normal_cholesky(rbeta[1], sigma);
  ry ~ multi_normal_cholesky(rz, sigma);
  ry ~ multi_normal_cholesky(rz[1], sigma);
  ry[1] ~ multi_normal_cholesky(rz, sigma);
  ry[1] ~ multi_normal_cholesky(rz[1], sigma);



  y ~ multi_normal_prec(beta, sigma);
  y ~ multi_normal_prec(beta[1], sigma);
  y[1] ~ multi_normal_prec(beta, sigma);
  y[1] ~ multi_normal_prec(beta[1], sigma);
  y ~ multi_normal_prec(z, sigma);
  y ~ multi_normal_prec(z[1], sigma);
  y[1] ~ multi_normal_prec(z, sigma);
  y[1] ~ multi_normal_prec(z[1], sigma);

  y ~ multi_normal_prec(rbeta, sigma);
  y ~ multi_normal_prec(rbeta[1], sigma);
  y[1] ~ multi_normal_prec(rbeta, sigma);
  y[1] ~ multi_normal_prec(rbeta[1], sigma);
  y ~ multi_normal_prec(rz, sigma);
  y ~ multi_normal_prec(rz[1], sigma);
  y[1] ~ multi_normal_prec(rz, sigma);
  y[1] ~ multi_normal_prec(rz[1], sigma);
  
  ry ~ multi_normal_prec(beta, sigma);
  ry ~ multi_normal_prec(beta[1], sigma);
  ry[1] ~ multi_normal_prec(beta, sigma);
  ry[1] ~ multi_normal_prec(beta[1], sigma);
  ry ~ multi_normal_prec(z, sigma);
  ry ~ multi_normal_prec(z[1], sigma);
  ry[1] ~ multi_normal_prec(z, sigma);
  ry[1] ~ multi_normal_prec(z[1], sigma);
  
  ry ~ multi_normal_prec(rbeta, sigma);
  ry ~ multi_normal_prec(rbeta[1], sigma);
  ry[1] ~ multi_normal_prec(rbeta, sigma);
  ry[1] ~ multi_normal_prec(rbeta[1], sigma);
  ry ~ multi_normal_prec(rz, sigma);
  ry ~ multi_normal_prec(rz[1], sigma);
  ry[1] ~ multi_normal_prec(rz, sigma);
  ry[1] ~ multi_normal_prec(rz[1], sigma);
  
  
  
  y ~ multi_student_t(10, beta, sigma);
  y ~ multi_student_t(10, beta[1], sigma);
  y[1] ~ multi_student_t(10, beta, sigma);
  y[1] ~ multi_student_t(10, beta[1], sigma);
  y ~ multi_student_t(10, z, sigma);
  y ~ multi_student_t(10, z[1], sigma);
  y[1] ~ multi_student_t(10, z, sigma);
  y[1] ~ multi_student_t(10, z[1], sigma);
  
  y ~ multi_student_t(10, rbeta, sigma);
  y ~ multi_student_t(10, rbeta[1], sigma);
  y[1] ~ multi_student_t(10, rbeta, sigma);
  y[1] ~ multi_student_t(10, rbeta[1], sigma);
  y ~ multi_student_t(10, rz, sigma);
  y ~ multi_student_t(10, rz[1], sigma);
  y[1] ~ multi_student_t(10, rz, sigma);
  y[1] ~ multi_student_t(10, rz[1], sigma);
  
  ry ~ multi_student_t(10, beta, sigma);
  ry ~ multi_student_t(10, beta[1], sigma);
  ry[1] ~ multi_student_t(10, beta, sigma);
  ry[1] ~ multi_student_t(10, beta[1], sigma);
  ry ~ multi_student_t(10, z, sigma);
  ry ~ multi_student_t(10, z[1], sigma);
  ry[1] ~ multi_student_t(10, z, sigma);
  ry[1] ~ multi_student_t(10, z[1], sigma);
  
  ry ~ multi_student_t(10, rbeta, sigma);
  ry ~ multi_student_t(10, rbeta[1], sigma);
  ry[1] ~ multi_student_t(10, rbeta, sigma);
  ry[1] ~ multi_student_t(10, rbeta[1], sigma);
  ry ~ multi_student_t(10, rz, sigma);
  ry ~ multi_student_t(10, rz[1], sigma);
  ry[1] ~ multi_student_t(10, rz, sigma);
  ry[1] ~ multi_student_t(10, rz[1], sigma);
}
