data {
  int<lower=0> N;
  array[N] int<lower=0, upper=1> y;
}
parameters {
  real<lower=0, upper=1> theta;
}
model {
  theta ~ beta(1, 1);
  y ~ bernoulli(theta);
}
generated quantities {
  real mu = theta;
  array[N] int y_rep;
  for (n in 1 : N) 
    y_rep[n] = bernoulli_rng(theta);
}

