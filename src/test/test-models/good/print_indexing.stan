data {
  array[2] int<lower=1> N;
  array[N[1]] int<lower=0> y_1;
  array[N[2]] int<lower=0> y_2;
}
parameters {
  real y;
}
model {
  y ~ normal(0, 1);
}

