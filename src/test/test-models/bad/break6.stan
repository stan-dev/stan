// tests right value passed through conditionals
parameters {
  real y;
}
model {
  if (1)
    break;
}
