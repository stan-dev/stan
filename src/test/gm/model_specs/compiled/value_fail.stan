parameters {
  real x;
}

model {
  lp__ <- lp__ + 1 / x;
}
