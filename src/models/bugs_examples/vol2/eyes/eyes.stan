// http://www.mrc-bsu.cam.ac.uk/bugs/winbugs/Vol2.pdf
// Page 11: Eyes: Normal Mixture Model 
// 
// not work yet, (not sure how to specify multivariate dsn: type of variables?)
// 
// from bugs example now, (have not looked at JAGS version yet) 
data {
  int(0,) N; 
  double y[N]; 
  vector(2) alpha;
} 
parameters {
  int(0,) z[N]; 
  double(0,) sigma;
  double(0,) theta;
  double lambda[2]; 
  vector(2) p;
} 
model {
  p ~ dirichlet(alpha); 
  theta ~ normal(0,1000);  // propto half normal because theta truncated
  lambda[1] ~ normal(0, 1e3); 
  lambda[2] <- lambda[1] + theta;
  // equiv: tau ~ gamma(); sigma <- 1 / sqrt(tau);
  1 / square(sigma) ~ gamma(1e-3, 1e-3);
  for (n in 1:N) {
    z[n] ~ categorical(p);
    y[n] ~ normal(lambda[z[n]], sigma);
  }
}

