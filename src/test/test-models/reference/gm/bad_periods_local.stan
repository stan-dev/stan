parameters {
  real z;
}
model {
  real x.y;
  z ~ normal(x.y,1);
}
