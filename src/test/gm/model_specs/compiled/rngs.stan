parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
generated quantities {
  int n;
  real z;
  
  n <- bernoulli_rng(0.5);
  // n <- bernoulli_logit_rng(0.0);

  z <- normal_rng(0,1);
}
