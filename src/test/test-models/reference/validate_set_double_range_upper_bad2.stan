data {
  vector[3] v;
  real<lower=2.9,upper=v> a;
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
