functions {
  real foo(real y) {
    return  1 ? y : 2;
  }
}
parameters {
  real y;
}
transformed parameters{
  real z;
  z = foo(y);
}
model {
  y ~ normal(0,1);
}
