transformed data {
  vector[2] mu = [2, 3]';
  cov_matrix[2] Sigma = [[1, 0.8], [0.8, 1]];
}
parameters {
  vector[2] y;	   
}
model {
  y ~ multi_normal(mu, Sigma);
}