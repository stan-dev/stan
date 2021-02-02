functions {
  array[] real sho(real t, array[] real y, array[] real theta,
                   array[] real x, array[] int x_int) {
    array[2] real dydt;
    dydt[1] = y[2];
    dydt[2] = -y[1] - theta[1] * y[2];
    return dydt;
  }
}
data {
  int<lower=1> T;
  array[2] real y0;
  real t0;
  array[T] real ts;
  array[1] real theta;
}
transformed data {
  array[0] real x;
  array[0] int x_int;
}
model {

}
generated quantities {
  array[T, 3] real y_hat;
  y_hat = integrate_ode_rk45(sho, y0, t0, ts, theta, x, x_int, 1e-10, 1e-10,
                             1e6);
  y_hat = integrate_ode_bdf(sho, y0, t0, ts, theta, x, x_int, 1e-10, 1e-10,
                            1e6);
  y_hat = integrate_ode_adams(sho, y0, t0, ts, theta, x, x_int, 1e-10, 1e-10,
                              1e6);
  for (t in 1 : T) {
    y_hat[t, 1] = y_hat[t, 1] + normal_rng(0, 0.1);
    y_hat[t, 2] = y_hat[t, 2] + normal_rng(0, 0.1);
    y_hat[t, 3] = y_hat[t, 3] + normal_rng(0, 0.1);
  }
}

