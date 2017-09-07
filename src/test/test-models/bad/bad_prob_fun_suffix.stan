functions {
  real foo(real theta, real sigma) {
    return -(theta / sigma)^2;
  }
}
parameters {
  real theta;
}
model {
  theta ~ foo(1.4);
}
