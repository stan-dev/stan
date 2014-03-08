transformed data {
  real x;
  vector[3] v;
  x <- v[1];
}
parameters {
  real y;
}
transformed parameters {
  real xt;
  vector[3] vt;
  xt <- vt[1];
}
model {
  y ~ normal(0,1);
}
