functions {
  real foo(real x, real xc, real[] theta, real[] x_r, int[] x_i) {
    return x^2;
  }
}
data {
  real x_r[2];
  int x_i[10];
}
transformed data {
  real theta_d[3];
  real int_foo1 = integrate_1d(foo, 0.2, 1.3, theta_d, x_r, x_i, 0.01);
}
parameters {
  real lb;
  real ub;
  real theta[3];
}
model {
  // all eight instantiations of (lb, ub, theta);
  real int_foo2 = integrate_1d(foo, 0.2, 1.3, theta, x_r, x_i, 0.01);
  real int_foo3 = integrate_1d(foo, lb, 1.3, theta, x_r, x_i, 0.01);
  real int_foo4 = integrate_1d(foo, 0.2, ub, theta, x_r, x_i, 0.01);
  real int_foo5 = integrate_1d(foo, lb, ub, theta, x_r, x_i, 0.01);

  // redundant test for int_foo6 given transformed data test
  real int_foo6 = integrate_1d(foo, 0.2, 1.3, theta_d, x_r, x_i, 0.01);
  real int_foo7 = integrate_1d(foo, lb, 1.3, theta_d, x_r, x_i, 0.01);
  real int_foo8 = integrate_1d(foo, 0.2, ub, theta_d, x_r, x_i, 0.01);
  real int_foo9 = integrate_1d(foo, lb, ub, theta_d, x_r, x_i, 0.01);
}
