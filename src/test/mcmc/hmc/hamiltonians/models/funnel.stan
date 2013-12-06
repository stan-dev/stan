transformed data {
  int N;
  N <- 10;
}

parameters {
  real v;
  vector[N] x;
}

model {
  v ~ normal(0, 3);
  x ~ normal(0, exp(v));
}