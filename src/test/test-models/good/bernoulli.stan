data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  real theta;
}
model {
  theta ~ normal(0, .5);
  y ~ normal(theta, 1);
}
