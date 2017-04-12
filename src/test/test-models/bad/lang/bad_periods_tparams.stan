transformed parameters {
  real x.;
  x. = 1.0;
}
model {
  2.0 ~ normal(x.,1);
}
