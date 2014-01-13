data {
  int<lower=1> N[2];
  int<lower=0> y_1[N[1]];
  int<lower=0> y_2[N[2]];
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
