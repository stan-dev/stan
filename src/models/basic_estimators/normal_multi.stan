data {
  int<lower=0> N; //response variable size
  int<lower=0> J; //number of observations
  vector[N] y[J];
  matrix[N, N] sigma;
}
transformed data {
  row_vector[N] ry[J];
  matrix[N, N] inv_sigma;
  for (n in 1:N)
    for (j in 1:J)
      ry[j][n] <- y[j][n];
  inv_sigma <- inverse_spd(sigma);
}
parameters {
  vector[N] beta[J];
  row_vector[N] rbeta[J];
}
model {
  for (n in 1:J)
    ry[n] ~ multi_normal(beta[n], sigma);
  y ~ multi_normal(beta, sigma);
  ry ~ multi_normal(beta[1], sigma);
  y[1] ~ multi_normal(beta, sigma);
  for (n in 1:J)
    y[n] ~ multi_normal(rbeta[n], sigma);
  y ~ multi_normal(rbeta, sigma);
  y ~ multi_normal(rbeta[1], sigma);
  ry[1] ~ multi_normal(rbeta, sigma);
  
  for (n in 1:J)
    ry[n] ~ multi_normal_cholesky(beta[n], sigma);
  y ~ multi_normal_cholesky(beta, sigma);
  ry ~ multi_normal_cholesky(beta[1], sigma);
  y[1] ~ multi_normal_cholesky(beta, sigma);
  for (n in 1:J)
    y[n] ~ multi_normal_cholesky(rbeta[n], sigma);
  y ~ multi_normal_cholesky(rbeta, sigma);
  y ~ multi_normal_cholesky(rbeta[1], sigma);
  ry[1] ~ multi_normal_cholesky(rbeta, sigma);
  
  for (n in 1:J)
    ry[n] ~ multi_normal_prec(beta[n], inv_sigma);
  y ~ multi_normal_prec(beta, inv_sigma);
  ry ~ multi_normal_prec(beta[1], inv_sigma);
  y[1] ~ multi_normal_prec(beta, inv_sigma);
  for (n in 1:J)
    y[n] ~ multi_normal_prec(rbeta[n], inv_sigma);
  y ~ multi_normal_prec(rbeta, inv_sigma);
  y ~ multi_normal_prec(rbeta[1], inv_sigma);
  ry[1] ~ multi_normal_prec(rbeta, inv_sigma);
}
