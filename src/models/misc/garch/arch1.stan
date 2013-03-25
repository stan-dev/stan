// ARCH(1) model

data {
  int<lower=0> T;   // measurement time indexes
  real r[T];        // return at time t
}
parameters {
  real mu;                       // average return
  real<lower=0> alpha0;          // noise intercept coefficient
  real<lower=0,upper=1> alpha1;  // noise slope coefficient
}
model {
  for (t in 2:T)
    r[t] ~ normal(mu, sqrt(alpha0 + alpha1 * pow(r[t-1] - mu,2)));
}
