# http://www.mrc-bsu.cam.ac.uk/bugs/winbugs/Vol2.pdf
# Page 23: Birats
## 
## not works yet for some multi_normal specification issue

data {
  int(0,) N;
  int(0,) T;
  double x[T];
  double xbar;
  double y[N,T];
  cov_matrix(2) Omega; 
}
parameters {
  vector(2)  beta[N];
  vector(2) mu_beta;
  double(0,) sigma_y;
  cov_matrix(2) Sigma_beta; 
}
derived parameters {
  double(0,) sigmasq_y;
  double(-1, 1) rho; 
  double alpha0; 
  sigmasq_y <- sigma_y * sigma_y;
  //rho <- Sigma_beta[1, 2] / sqrt(Sigma_beta[1, 1] * Sigma_beta[2, 2]);
  //alpha0 <- mu_beta[1] - mu_beta[2] * xbar; 
}
model {
  sigmasq_y ~ inv_gamma(0.001, 0.001);
  mu_beta[1] ~ normal(0, 100); 
  mu_beta[2] ~ normal(0, 100); 
  Sigma_beta ~ inv_wishart(2, Omega); 
  for (n in 1:N) 
    beta[n] ~ multi_normal(mu_beta, Sigma_beta);
  for (n in 1:N)
    for (t in 1:T) 
      y[n,t] ~ normal(beta[n, 1]+ beta[n, 2] * (x[t] - xbar), sigma_y);
}
