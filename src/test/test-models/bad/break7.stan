// tests right value passed through else if in conditional
parameters {
  real y;
}
model {
  int z;
  if (1)
    z = 10;
  else if (0)
    break;
}
