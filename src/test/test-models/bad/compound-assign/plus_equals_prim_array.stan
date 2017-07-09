generated quantities {
  real x[3] = {1.0, 2.0, 3.0};
  real y[3] = {4.0, 5.0, 6.0};
  x[1] += y[1];
  x += y;
}
