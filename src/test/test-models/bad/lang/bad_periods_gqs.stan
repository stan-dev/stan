parameters {
  real z;
}
model {
  z ~ normal(0,1);
}
generated quantities {
  real x.y;
  x.y = z * 2;
}
