data {
  int<lower=2> K;
  ordered[K] v;
}
transformed data {
  vector<lower=0>[K-1] a1;
  real<lower=0> a2[K-1];

  a1 <- tail(v, K-1) - head(v, K-1);
  a2 <- tail(v, K-1) - head(v, K-1);
}
parameters {
  real x;
}
transformed parameters {
}
model {
  x ~ normal(0, 1);
}
generated quantities {
}
