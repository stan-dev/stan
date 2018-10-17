data {
  int N;
  vector[N] y;
}
transformed data {
  vector[N] abs_y;
  abs_y = abs(y);
}
parameters {
  real theta;
}
model {
  y ~ normal(theta, 1);
}
