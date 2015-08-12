transformed data {
  real x;
  real y;
  real z;
  int n;

  z <- if_else(n,x,y);

  z <- binomial_coefficient_log(x,y);
}
model {
}
