data {
  int<lower=0> T; 
  real y[T];
  real<lower=0> sigma1; 
}

parameters {
  real mu; 
  real<lower=0> alpha0;          
  real<lower=0, upper=1> alpha1;
  real<lower=0, upper=(1-alpha1)> beta1; 
}

model {
  real sigma[T];
  sigma[1] = sigma1;
  for (t in 2:T)
    sigma[t] = sqrt(  alpha0
                    + alpha1 * square(y[t - 1] - mu)
                    + beta1 * square(sigma[t - 1]));

  y ~ normal(mu, sigma);
}
