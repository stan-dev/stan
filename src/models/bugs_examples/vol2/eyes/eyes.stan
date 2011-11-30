// http://www.mrc-bsu.cam.ac.uk/bugs/winbugs/Vol2.pdf
// Page 11: Eyes: Normal Mixture Model 
// 
// not work yet, (not sure how to specify multivariate dsn: type of variables?)
// 
// from bugs example now, (have not looked at JAGS version yet) 
data {
  int(0,) N; 
  double y[N]; 
  double alpha[2];
} 
parameters {
  int(0,) T[N]; 
  double(0,) sigma;
  double(0,) theta;
  double lambda[2]; 
  // how to specify P, which has prior of Dirichlet dsn? 
  matrix(2,1) P; 
} 

derived parameters {
  double(0,) tau; 
  tau <- 1 / (sigma * sigma); 
} 
model {
  for (i in 1:N) {
    y[i] ~ normal(lambda[T[i]], sigma);
    T[i] ~ categorical(P);
  }
  P ~ dirichlet(alpha); 
  lambda[2] <- lambda[1] + theta;
  theta ~ normal_trunc_l(0, 1000, 0); 
  lambda[1] ~ normal(0.0, 1000); 
  tau ~ gamma(0.001, 0.001); 
}

