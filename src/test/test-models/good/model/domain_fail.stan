parameters {
  real<lower=0> x;
}
model {
  target += -sqrt(-x);
}

