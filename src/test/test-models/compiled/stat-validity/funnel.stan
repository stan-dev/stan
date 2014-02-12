transformed data {
  int N;
  N <- 5;
}

parameters {
  real v;
  vector[N] x;
}

model {
  v ~ normal(0, 3);
  x ~ normal(0, exp(v/2));
}

generated quantities {
  real v2;
  v2 <- v * v;
}
