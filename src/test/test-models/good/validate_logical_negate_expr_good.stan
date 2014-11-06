transformed data {
  int n;
  real x;
  n <- !n;
  x <- !x;
}
parameters {
  real y;
}
transformed parameters {
  real xt;
  xt <- !xt;
}
model {
  y ~ normal(0,1);
}
