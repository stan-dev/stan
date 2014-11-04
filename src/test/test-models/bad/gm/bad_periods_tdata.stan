transformed data {
  real x.y;
}
parameters {
  real z;
}
model {
  z ~ normal(x.y,1);
}
