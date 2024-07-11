data {
  int<lower=1> N;
}
parameters {
  vector[N] y;
}
model {
  target += y;
}
