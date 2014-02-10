transformed data {
  int N;
  N <- 25;
}

parameters {
  real v;
  vector[N] x;
}

transformed parameters {
  real v2;
  v2 <- v * v;
}

model {
  v ~ normal(0, 3);
  x ~ normal(0, exp(v));
}