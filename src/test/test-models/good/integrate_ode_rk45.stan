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
  array[2] real y0_d;
  real t0;
  array[T] real ts;
  array[1] real theta_d;
  array[0] real x;
  array[0] int x_int;
}
parameters {
  array[2] real y0_p;
  array[1] real theta_p;
  array[T] real ts_p;
  real t0_p;
}
model {
  array[T, 2] real y_hat;
  y_hat = integrate_ode_rk45(sho, y0_d, t0, ts, theta_d, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_d, t0, ts, theta_p, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_p, t0, ts, theta_d, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_p, t0, ts, theta_p, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_d, t0_p, ts, theta_d, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_d, t0_p, ts, theta_p, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_p, t0_p, ts, theta_d, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_p, t0_p, ts, theta_p, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_d, t0, ts_p, theta_d, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_d, t0, ts_p, theta_p, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_p, t0, ts_p, theta_d, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_p, t0, ts_p, theta_p, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_d, t0_p, ts_p, theta_d, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_d, t0_p, ts_p, theta_p, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_p, t0_p, ts_p, theta_d, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_p, t0_p, ts_p, theta_p, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_d, t0, ts, theta_d, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_d, t0, ts, theta_p, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_p, t0, ts, theta_d, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_p, t0, ts, theta_p, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_d, t0_p, ts, theta_d, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_d, t0_p, ts, theta_p, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_p, t0_p, ts, theta_d, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_p, t0_p, ts, theta_p, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_d, t0, ts_p, theta_d, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_d, t0, ts_p, theta_p, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_p, t0, ts_p, theta_d, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_p, t0, ts_p, theta_p, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_d, t0_p, ts_p, theta_d, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_d, t0_p, ts_p, theta_p, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_p, t0_p, ts_p, theta_d, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_p, t0_p, ts_p, theta_p, x, x_int, 1e-10,
                             1e-10, 1e8);
}
generated quantities {
  array[T, 2] real y_hat;
  y_hat = integrate_ode_rk45(sho, y0_d, t0, ts, theta_d, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_d, t0, ts, theta_p, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_p, t0, ts, theta_d, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_p, t0, ts, theta_p, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_d, t0_p, ts, theta_d, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_d, t0_p, ts, theta_p, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_p, t0_p, ts, theta_d, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_p, t0_p, ts, theta_p, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_d, t0, ts_p, theta_d, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_d, t0, ts_p, theta_p, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_p, t0, ts_p, theta_d, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_p, t0, ts_p, theta_p, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_d, t0_p, ts_p, theta_d, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_d, t0_p, ts_p, theta_p, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_p, t0_p, ts_p, theta_d, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_p, t0_p, ts_p, theta_p, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_d, t0, ts, theta_d, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_d, t0, ts, theta_p, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_p, t0, ts, theta_d, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_p, t0, ts, theta_p, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_d, t0_p, ts, theta_d, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_d, t0_p, ts, theta_p, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_p, t0_p, ts, theta_d, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_p, t0_p, ts, theta_p, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_d, t0, ts_p, theta_d, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_d, t0, ts_p, theta_p, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_p, t0, ts_p, theta_d, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_p, t0, ts_p, theta_p, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_d, t0_p, ts_p, theta_d, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_d, t0_p, ts_p, theta_p, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_p, t0_p, ts_p, theta_d, x, x_int, 1e-10,
                             1e-10, 1e8);
  y_hat = integrate_ode_rk45(sho, y0_p, t0_p, ts_p, theta_p, x, x_int, 1e-10,
                             1e-10, 1e8);
}

