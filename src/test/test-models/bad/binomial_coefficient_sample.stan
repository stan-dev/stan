data {
  real y;
}
model {
  y ~ binomial_coefficient(5);
}
