# http://www.mrc-bsu.cam.ac.uk/bugs/winbugs/Vol1.pdf
# Page 3: Rats
data {
  int(0,) N;
  int(0,) T;
//  double x[T];
//  double xbar;
  double Y[N,T];
}
parameters {
  double mu;
  double(0,) sigma;
/*  double mu[N,T];
  double(0,) sigma_c;
  double alpha[N];
  double beta[N];
  double alpha_c;
  double(0,) alpha_sigma;
  double beta_c;
  double(0,) beta_sigma;
  double alpha0;*/
}
model {
  for (i in 1:N) {
    for (j in 1:T) {
      Y[i,j] ~ normal(mu, sigma);
    }
  } 
/*  sigma_c ~ inv_gamma (0.001, 0.001);
  for (i in 1:N) {
    for (j in 1:T) {
      Y [i,j] ~ normal(mu[i,j], sigma_c);
      mu[i,j] <- alpha[i] + beta[i] * (x[j] - xbar);
    }
    alpha[i] ~ normal(alpha_c, alpha_sigma);
    beta[i] ~ normal(beta_c, beta_sigma);
  } 
  sigma_c ~ inv_gamma(0.001,0.001);
  alpha_c ~ normal(0.0,1.0E6);
  alpha_sigma ~ inv_gamma(0.001,0.001);
  beta_c ~ normal(0.0,1.0E6);
  beta_sigma ~ inv_gamma(0.001,0.001);
  alpha0 <- alpha_c - xbar * beta_c;*/
}
