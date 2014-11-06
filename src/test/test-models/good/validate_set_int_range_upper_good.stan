data {
  int<upper=1> a;
  int<lower=1,upper=3> b;
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
