data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N,K] X;
  vector[N] y;
}
parameters {
  vector[K] b;
  real<lower=0> sigma;
}
model {
  y ~ normal(X*b, sigma);
}
