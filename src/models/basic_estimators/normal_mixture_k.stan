data {
  int<lower=1> K;
  int<lower=1> N;
  real y[N];
}
parameters {
  simplex[K] theta;
  real mu[K];
  real<lower=0,upper=10> sigma[K];
}
model {
  real ps[K];              
  mu ~ normal(0,10);
  for (n in 1:N) {
    for (k in 1:K)
      ps[k] <- log(theta[k]) 
               + normal_log(y[n],mu[k],sigma[k]);
    lp__ <- lp__ + log_sum_exp(ps);    
  }
}
