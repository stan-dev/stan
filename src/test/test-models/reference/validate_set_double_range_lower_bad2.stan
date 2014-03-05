data {
  vector[3] v;
  real<lower=v,upper=2.9> a;
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
