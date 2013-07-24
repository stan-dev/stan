// http://www.mrc-bsu.cam.ac.uk/bugs/winbugs/Vol2.pdf
// Page 11: Eyes: Normal Mixture Model 
// 

// works using the mixture distribution form instead of 
// latent indicators 

data {
  int<lower=0> N; 
  real y[N]; 
} 
parameters {
  real<lower=0> sigmasq;
  real<lower=0> theta;
  real lambda_1; 
  real<lower=0,upper=1> p1; 
} 
transformed parameters {
    real lambda[2];
    real sigma; 
    sigma <- sqrt(sigmasq); 
    lambda[1] <- lambda_1;
    lambda[2] <- lambda[1] + theta;
}

model {
  theta ~ normal(0, 100); 
  lambda_1 ~ normal(0, 1e3); 
  sigmasq ~ inv_gamma(1e-3, 1e-3); 
  for (n in 1:N) {
    lp__ <- lp__ + 
            log_sum_exp(log(p1) + normal_log(y[n], lambda[1], sigma), 
                        log(1 - p1) + normal_log(y[n], lambda[2], sigma)); 
  }
} 

