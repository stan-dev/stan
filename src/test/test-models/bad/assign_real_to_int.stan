parameters {
  real y;
}
model { 
  y ~ normal(0, 1);
}
generated quantities {
  int a;
  real b;
  b <- 3.2;
  a <- b;
}
