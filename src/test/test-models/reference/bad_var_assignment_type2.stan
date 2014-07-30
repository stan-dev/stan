data {
  real y[3,3];
}
transformed data {
  real z[5];
  z <- y;
}
parameters {
  real x;
}
model {
  x ~ normal(0, 1);
}

