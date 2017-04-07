transformed data {
  int tx;
  real ty;
  row_vector[6] twa2[2,2];
  tx = twa2 ? 2 : 3;   // BAD
}
parameters {
  real py;
}
model {
  py ~ normal(0,1);
}
