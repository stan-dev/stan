// normal mixture, unknown proportion and means, known variance
// p(y|mu,theta) = theta * Normal(y|mu[1],1) + (1-theta) * Normal(y|mu[2],1);

data {
  int<lower=0>  N;
  real y[N];
}
parameters {
  real<lower=0,upper=1> theta;
  real mu[2];
}
transformed parameters {
  real log_theta;
  real log_one_minus_theta;

  log_theta <- log(theta);
  log_one_minus_theta <- log(1.0 - theta);
}
model {
  theta ~ uniform(0,1); // equivalently, ~ beta(1,1);
  for (k in 1:2)
    mu[k] ~ normal(0,10);
  for (n in 1:N)
    increment_log_prob(log_sum_exp(log_theta
                                     + normal_log(y[n],mu[1],1.0),
                                   log_one_minus_theta 
                                     + normal_log(y[n],mu[2],1.0)));
}
