data {
  vector[3] v;
  real<lower=v> a;
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
