// Predict from Gaussian Process
// Fixed covar function: eta_sq=1, rho_sq=1, sigma_sq=0.1

data {
  int<lower=0> N1;     
  vector[N1] x1; 
  vector[N1] y1;
  int<lower=0> N2;
  vector[N2] x2;
}
transformed data {
  vector[N] mu;
  int<lower=1> N;
  vector[N] x;
  for (i in 1:N) mu[i] <- 0;
  for (n in 1:N1) x[n] <- x1[n];
  for (n in 1:N2) x[N1 + n] <- x2[n];
}
parameters {
  real y2[N];
}
model {
  vector[N] y;
  cov_matrix[N] Sigma;
  for (n in 1:N1) y[n] <- y1[n];
  for (n in 1:N2) y[N1 + n] <- y2[n];
  for (i in 1:N) 
    for (j in 1:N)
      Sigma[i,j] <- exp(-pow(x[i] - x[j],2)) + if_else(i==j, 0.1, 0.0);
  y ~ multi_normal_cholesky(mu,Sigma);
}
