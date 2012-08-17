// http://www.mrc-bsu.cam.ac.uk/bugs/winbugs/Vol2.pdf
// Page 11: Eyes: Normal Mixture Model 
// 

// FIXME vI: use beta-bernoulli rather than dirichlet-multinomial (done), no bernoulli now 
// FIXME vII: marginalize out z[N] (done) 


// works using the mixture distribution form instead of 
// latent indicators 

data {
  int<lower=0> N; 
  real y[N]; 
//  vector[2] alpha;
} 
parameters {
  // int<lower=0> z[N]; 
  real<lower=0> sigmasq;
  real<lower=0> theta;
  real lambda_1; 
  // vector[2] p;
  real<lower=0,upper= 1> p1; 
} 
transformed parameters {
    real lambda[2];
    real sigma; 
    sigma <- sqrt(sigmasq); 
    lambda[1] <- lambda_1;
    lambda[2] <- lambda[1] + theta;
}
/* 
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
*/

model {
  // p1 ~ beta(alpha[1], alpha[2]); 
  p1 ~ beta(1, 1); 
  theta ~ normal(0, 100); 
  lambda_1 ~ normal(0, 1e3); 
  sigmasq ~ inv_gamma(1e-3, 1e-3); 
  for (n in 1:N) {
    lp__ <- lp__ + 
            log_sum_exp(log(p1) + normal_log(y[n], lambda[1], sigma), 
                        log(1 - p1) + normal_log(y[n], lambda[2], sigma)); 
  }
} 

