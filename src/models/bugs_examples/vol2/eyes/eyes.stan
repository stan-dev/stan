// http://www.mrc-bsu.cam.ac.uk/bugs/winbugs/Vol2.pdf
// Page 11: Eyes: Normal Mixture Model 
// 
// not work yet, (not sure how to specify multivariate dsn: type of variables?)
// 
// from bugs example now, (have not looked at JAGS version yet) 

// FIXME vI: use beta-bernoulli rather than dirichlet-multinomial
// FIXME vII: marginalize out z[N]

data {
  int(0,) N; 
  double y[N]; 
  vector(2) alpha;
} 
parameters {
  int(0,) z[N]; 
  double(0,) sigmasq;
  double(0,) theta;
  double lambda_1; 
  vector(2) p;
} 
derived parameters {
    double lambda[2];
    lambda[1] <- lambda_1;
    lambda[2] <- lambda[1] + theta;
}
model {
  p ~ dirichlet(alpha); 
  theta ~ normal(0, 1000);  // propto half normal because theta truncated
  lambda_1 ~ normal(0, 1e3); 
  sigmasq ~ inv_gamma(1e-3, 1e-3); 
  for (n in 1:N) {
    z[n] ~ categorical(p);
    y[n] ~ normal(lambda[z[n]], sqrt(sigmasq)); 
  }
}

