transformed data {
  vector[5] y;
  real z[5];
  z <- y;
}
parameters {
  real x;
}
model {
  x ~ normal(0, 1);
}
