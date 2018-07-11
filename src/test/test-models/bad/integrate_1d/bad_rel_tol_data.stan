functions {
  real foo(real x, real xc, real[] theta, real[] x_r, int[] x_i) {
    return x^2;
  }
}
data {
  real x_r[2];
  int x_i[10];
}
parameters {
  real theta[3];
  real tol;
}
transformed parameters {
  real int_foo1 = integrate_1d(foo, 0.2, 1.3, theta, x_r, x_i, tol);
}
