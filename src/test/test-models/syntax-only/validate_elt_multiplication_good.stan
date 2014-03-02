transformed data {
  matrix[3,3] m;
  vector[3] v;
  row_vector[3] rv;
  m <- m .* m;
  v <- v .* v;
  rv <- rv .* rv;
}
parameters {
  real y;
}
transformed parameters {
  matrix[3,3] mt;
  vector[3] vt;
  row_vector[3] rvt;
  mt <- mt .* mt;
  vt <- vt .* vt;
  rvt <- rvt .* rvt;
}
model {
  y ~ normal(0,1);
}
