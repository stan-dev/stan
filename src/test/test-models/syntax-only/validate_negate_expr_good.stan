transformed data {
  int n;
  real x;
  matrix[3,3] m;
  vector[3] v;
  row_vector[3] rv;
  n <- -n;
  x <- -x;
  m <- -m;
  v <- -v;
  rv <- -rv;
}
parameters {
  real y;
}
transformed parameters {
  real xt;
  matrix[3,3] mt;
  vector[3] vt;
  row_vector[3] rvt;
  xt <- -xt;
  mt <- -mt;
  vt <- -vt;
  rvt <- -rvt;
}
model {
  y ~ normal(0,1);
}
