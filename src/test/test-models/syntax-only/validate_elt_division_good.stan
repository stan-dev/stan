transformed data {
  matrix[3,3] m;
  vector[3] v;
  row_vector[3] rv;
  real c;
  m <- m ./ m;
  m <- m ./ c;
  m <- c ./ m;
  v <- v ./ v;
  v <- c ./ v;
  v <- v ./ c;
  rv <- rv ./ rv;
  rv <- c ./ rv;
  rv <- rv ./ c;
}
parameters {
  real y;
}
transformed parameters {
  matrix[3,3] mt;
  vector[3] vt;
  row_vector[3] rvt;
  real ct;
  mt <- mt ./ mt;
  mt <- mt ./ ct;
  mt <- ct ./ mt;
  vt <- vt ./ vt;
  vt <- ct ./ vt;
  vt <- vt ./ ct;
  rvt <- rvt ./ rvt;
  rvt <- ct ./ rvt;
  rvt <- rvt ./ ct;
}
model {
  y ~ normal(0,1);
}
