parameters {
  real mu_lpdf;
}
model {
  target += mu_lpdf;
}
