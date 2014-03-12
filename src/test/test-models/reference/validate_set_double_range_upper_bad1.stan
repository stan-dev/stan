data {
  vector[3] v;
  real<upper=v> a;
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
