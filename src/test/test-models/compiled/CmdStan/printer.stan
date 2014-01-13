transformed data {
  real x;
  x <- 2;
  print("x=",x);
}
parameters {
  real y;
}
transformed parameters {
  real z;
  z <- y * y;
  print("z=",z);
}
model {
  y ~ normal(0,1);
  print("y=",y);
}
generated quantities {
  real w;
  w <- z / 2;
  print("w=",w);
}
