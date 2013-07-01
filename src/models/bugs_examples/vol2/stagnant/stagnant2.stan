data {
  int<lower=0> N;
  real x[N];
  real Y[N];
}
parameters {
  real<lower=0> sigma;
  real<lower=0> alpha;
  real beta[2];
  real<lower=min(x),upper=max(x)> x_change;
}
model {
  real mu[N];

  alpha  ~ normal(0,5);
  beta ~ normal(0,5);
  sigma ~ cauchy(0,5);

  for (n in 1:N)
    mu[n] <- alpha 
      + if_else(x[n] < x_change, beta[1], beta[2]) * (x[n] - x_change);

  Y ~ normal(mu,sigma);
}

