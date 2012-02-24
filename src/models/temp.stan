data {
  real(0,) x;
}
transformed data {
  real(0,1) y;
}
parameters {
  real(0,) sigma;
}
transformed parameters {
  real(0,) sigma_sq;
}
model {
   real z;
}
generated quantities {
  real(0,1) w;
}
