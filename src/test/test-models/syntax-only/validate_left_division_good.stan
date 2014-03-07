transformed data {
  matrix[3,3] m;
  vector[3] v;
  m <- m \ m;
  v <- m \ v;
}
parameters {
  real y;
}
transformed parameters {
  matrix[3,3] mt;
  vector[3] vt;
  mt <- mt \ mt;
  vt <- mt \ vt;
}
model {
  y ~ normal(0,1);
}
