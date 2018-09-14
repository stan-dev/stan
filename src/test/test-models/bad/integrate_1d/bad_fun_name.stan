data {
  real x_r[2];
  int x_i[10];
}
transformed data {
  real theta_d[3];
  real int_foo1 = integrate_1d(normal_rng, 0.2, 1.3, theta_d, x_r, x_i, 0.01);
}
