data {
  real a;
  vector[3] b;
  array[7] real c;
  array[8, 9] real d;
}
parameters {
  real e;
  vector[3] f;
  array[7] real g;
  array[8, 9] real h;
}
model {
  target += -e ^ 2 / 2;
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

