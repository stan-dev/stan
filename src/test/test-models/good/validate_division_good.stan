transformed data {
  real u;
  matrix[3,3] m;
  row_vector[3] v;
  u <- 2.1 / 3;
  u <- 2 / 3.1;
  u <- 2.1 / 3.1;
  m <- m / m;
  v <- v / m;
}
parameters {
  real y;
}
transformed parameters {
  real xt;
  real ut;
  matrix[3,3] mt;
  row_vector[3] vt;
  xt <- 2 / 3;
  ut <- 2.1 / 3;
  ut <- 2 / 3.1;
  ut <- 2.1 / 3.1;
  mt <- mt / mt;
  vt <- vt / mt;
}
model {
  y ~ normal(0,1);
}
