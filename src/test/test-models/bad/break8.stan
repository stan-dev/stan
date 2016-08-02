// tests right value passed through else in conditional
parameters {
  real y;
}
model {
  int z;
  if (1)
    z = 10;
  else if (0)
    z = 5;
  else
    break;
}
