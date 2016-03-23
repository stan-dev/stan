data {
  real a;
  vector[3] b;
  real c[7];
  real d[8, 9];
}
parameters {
  real e;
  vector[3] f;
  real g[7];
  real h[8, 9];
}
model {
  target += -e^2 / 2;
  target += a;
  target += b;
  target += b;
  target += c;
  target += d;
  target += e;
  target += f;
  target += g;
  target += h;
}
