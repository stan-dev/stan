data {
  int<lower=1> N;
  int<lower=0> y[N];
}
parameters {
  real<lower=0> alpha;
  real<lower=0,upper=1> p_success;
}
transformed parameters {
   real<lower=0.0> beta;
   beta <- p_success / (1.0 - p_success);
}
model { 
  for (i in 1:N)
    y[i] ~ neg_binomial(alpha, beta);
}
