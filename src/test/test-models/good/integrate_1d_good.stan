functions {
  real foo(real x, real xc, array[] real theta, array[] real x_r,
           array[] int x_i) {
    return x ^ 2;
  }
}
data {
  array[2] real x_r;
  array[10] int x_i;
}
transformed data {
  array[3] real theta_d;
  real int_foo1 = integrate_1d(foo, 0.2, 1.3, theta_d, x_r, x_i, 0.01);
}
parameters {
  real lb;
  real ub;
  array[3] real theta;
}
model {
  real int_foo2 = integrate_1d(foo, 0.2, 1.3, theta, x_r, x_i, 0.01);
  real int_foo3 = integrate_1d(foo, lb, 1.3, theta, x_r, x_i, 0.01);
  real int_foo4 = integrate_1d(foo, 0.2, ub, theta, x_r, x_i, 0.01);
  real int_foo5 = integrate_1d(foo, lb, ub, theta, x_r, x_i, 0.01);
  real int_foo6 = integrate_1d(foo, 0.2, 1.3, theta_d, x_r, x_i, 0.01);
  real int_foo7 = integrate_1d(foo, lb, 1.3, theta_d, x_r, x_i, 0.01);
  real int_foo8 = integrate_1d(foo, 0.2, ub, theta_d, x_r, x_i, 0.01);
  real int_foo9 = integrate_1d(foo, lb, ub, theta_d, x_r, x_i, 0.01);
}

