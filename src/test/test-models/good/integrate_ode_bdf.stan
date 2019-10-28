functions {
  real[] sho(real t,
             real[] y, 
             real[] theta,
             real[] x,
             int[] x_int) {
    real dydt[2];
    dydt[1] = y[2];
    dydt[2] = -y[1] - theta[1] * y[2];
    return dydt;
  }
}
data {
  int<lower=1> T;
  real y0_d[2];
  real t0;
  real ts[T];
  real theta_d[1];
  real x[0];
  int x_int[0];
}
parameters {
  real y0_p[2];
  real theta_p[1];
  real ts_p[T];
  real t0_p;
}
model {
  real y_hat[T,2];
  y_hat = integrate_ode_bdf(sho, y0_d, t0, ts, theta_d, x, x_int);
  y_hat = integrate_ode_bdf(sho, y0_d, t0, ts, theta_p, x, x_int);
  y_hat = integrate_ode_bdf(sho, y0_p, t0, ts, theta_d, x, x_int);
  y_hat = integrate_ode_bdf(sho, y0_p, t0, ts, theta_p, x, x_int);
  // let t0 be a parameter
  y_hat = integrate_ode_bdf(sho, y0_d, t0_p, ts, theta_d, x, x_int);
  y_hat = integrate_ode_bdf(sho, y0_d, t0_p, ts, theta_p, x, x_int);
  y_hat = integrate_ode_bdf(sho, y0_p, t0_p, ts, theta_d, x, x_int);
  y_hat = integrate_ode_bdf(sho, y0_p, t0_p, ts, theta_p, x, x_int);
  // let ts be a parameter
  y_hat = integrate_ode_bdf(sho, y0_d, t0, ts_p, theta_d, x, x_int);
  y_hat = integrate_ode_bdf(sho, y0_d, t0, ts_p, theta_p, x, x_int);
  y_hat = integrate_ode_bdf(sho, y0_p, t0, ts_p, theta_d, x, x_int);
  y_hat = integrate_ode_bdf(sho, y0_p, t0, ts_p, theta_p, x, x_int);
  // let both time argument be parameters
  y_hat = integrate_ode_bdf(sho, y0_d, t0_p, ts_p, theta_d, x, x_int);
  y_hat = integrate_ode_bdf(sho, y0_d, t0_p, ts_p, theta_p, x, x_int);
  y_hat = integrate_ode_bdf(sho, y0_p, t0_p, ts_p, theta_d, x, x_int);
  y_hat = integrate_ode_bdf(sho, y0_p, t0_p, ts_p, theta_p, x, x_int);

  y_hat = integrate_ode_bdf(sho, y0_d, t0, ts, theta_d, x, x_int, 1e-10, 1e-10, 1e8);
  y_hat = integrate_ode_bdf(sho, y0_d, t0, ts, theta_p, x, x_int, 1e-10, 1e-10, 1e8);
  y_hat = integrate_ode_bdf(sho, y0_p, t0, ts, theta_d, x, x_int, 1e-10, 1e-10, 1e8);
  y_hat = integrate_ode_bdf(sho, y0_p, t0, ts, theta_p, x, x_int, 1e-10, 1e-10, 1e8);
  // let t0 be a parameter
  y_hat = integrate_ode_bdf(sho, y0_d, t0_p, ts, theta_d, x, x_int, 1e-10, 1e-10, 1e8);
  y_hat = integrate_ode_bdf(sho, y0_d, t0_p, ts, theta_p, x, x_int, 1e-10, 1e-10, 1e8);
  y_hat = integrate_ode_bdf(sho, y0_p, t0_p, ts, theta_d, x, x_int, 1e-10, 1e-10, 1e8);
  y_hat = integrate_ode_bdf(sho, y0_p, t0_p, ts, theta_p, x, x_int, 1e-10, 1e-10, 1e8);
  // let ts be a parameter
  y_hat = integrate_ode_bdf(sho, y0_d, t0, ts_p, theta_d, x, x_int, 1e-10, 1e-10, 1e8);
  y_hat = integrate_ode_bdf(sho, y0_d, t0, ts_p, theta_p, x, x_int, 1e-10, 1e-10, 1e8);
  y_hat = integrate_ode_bdf(sho, y0_p, t0, ts_p, theta_d, x, x_int, 1e-10, 1e-10, 1e8);
  y_hat = integrate_ode_bdf(sho, y0_p, t0, ts_p, theta_p, x, x_int, 1e-10, 1e-10, 1e8);
  // let both time argument be parameters
  y_hat = integrate_ode_bdf(sho, y0_d, t0_p, ts_p, theta_d, x, x_int, 1e-10, 1e-10, 1e8);
  y_hat = integrate_ode_bdf(sho, y0_d, t0_p, ts_p, theta_p, x, x_int, 1e-10, 1e-10, 1e8);
  y_hat = integrate_ode_bdf(sho, y0_p, t0_p, ts_p, theta_d, x, x_int, 1e-10, 1e-10, 1e8);
  y_hat = integrate_ode_bdf(sho, y0_p, t0_p, ts_p, theta_p, x, x_int, 1e-10, 1e-10, 1e8);
}
generated quantities {
  real y_hat[T,2];
  y_hat = integrate_ode_bdf(sho, y0_d, t0, ts, theta_d, x, x_int);
  y_hat = integrate_ode_bdf(sho, y0_d, t0, ts, theta_p, x, x_int);
  y_hat = integrate_ode_bdf(sho, y0_p, t0, ts, theta_d, x, x_int);
  y_hat = integrate_ode_bdf(sho, y0_p, t0, ts, theta_p, x, x_int);
  // let t0 be a parameter
  y_hat = integrate_ode_bdf(sho, y0_d, t0_p, ts, theta_d, x, x_int);
  y_hat = integrate_ode_bdf(sho, y0_d, t0_p, ts, theta_p, x, x_int);
  y_hat = integrate_ode_bdf(sho, y0_p, t0_p, ts, theta_d, x, x_int);
  y_hat = integrate_ode_bdf(sho, y0_p, t0_p, ts, theta_p, x, x_int);
  // let ts be a parameter
  y_hat = integrate_ode_bdf(sho, y0_d, t0, ts_p, theta_d, x, x_int);
  y_hat = integrate_ode_bdf(sho, y0_d, t0, ts_p, theta_p, x, x_int);
  y_hat = integrate_ode_bdf(sho, y0_p, t0, ts_p, theta_d, x, x_int);
  y_hat = integrate_ode_bdf(sho, y0_p, t0, ts_p, theta_p, x, x_int);
  // let both time argument be parameters
  y_hat = integrate_ode_bdf(sho, y0_d, t0_p, ts_p, theta_d, x, x_int);
  y_hat = integrate_ode_bdf(sho, y0_d, t0_p, ts_p, theta_p, x, x_int);
  y_hat = integrate_ode_bdf(sho, y0_p, t0_p, ts_p, theta_d, x, x_int);
  y_hat = integrate_ode_bdf(sho, y0_p, t0_p, ts_p, theta_p, x, x_int);

  y_hat = integrate_ode_bdf(sho, y0_d, t0, ts, theta_d, x, x_int, 1e-10, 1e-10, 1e8);
  y_hat = integrate_ode_bdf(sho, y0_d, t0, ts, theta_p, x, x_int, 1e-10, 1e-10, 1e8);
  y_hat = integrate_ode_bdf(sho, y0_p, t0, ts, theta_d, x, x_int, 1e-10, 1e-10, 1e8);
  y_hat = integrate_ode_bdf(sho, y0_p, t0, ts, theta_p, x, x_int, 1e-10, 1e-10, 1e8);
  // let t0 be a parameter
  y_hat = integrate_ode_bdf(sho, y0_d, t0_p, ts, theta_d, x, x_int, 1e-10, 1e-10, 1e8);
  y_hat = integrate_ode_bdf(sho, y0_d, t0_p, ts, theta_p, x, x_int, 1e-10, 1e-10, 1e8);
  y_hat = integrate_ode_bdf(sho, y0_p, t0_p, ts, theta_d, x, x_int, 1e-10, 1e-10, 1e8);
  y_hat = integrate_ode_bdf(sho, y0_p, t0_p, ts, theta_p, x, x_int, 1e-10, 1e-10, 1e8);
  // let ts be a parameter
  y_hat = integrate_ode_bdf(sho, y0_d, t0, ts_p, theta_d, x, x_int, 1e-10, 1e-10, 1e8);
  y_hat = integrate_ode_bdf(sho, y0_d, t0, ts_p, theta_p, x, x_int, 1e-10, 1e-10, 1e8);
  y_hat = integrate_ode_bdf(sho, y0_p, t0, ts_p, theta_d, x, x_int, 1e-10, 1e-10, 1e8);
  y_hat = integrate_ode_bdf(sho, y0_p, t0, ts_p, theta_p, x, x_int, 1e-10, 1e-10, 1e8);
  // let both time argument be parameters
  y_hat = integrate_ode_bdf(sho, y0_d, t0_p, ts_p, theta_d, x, x_int, 1e-10, 1e-10, 1e8);
  y_hat = integrate_ode_bdf(sho, y0_d, t0_p, ts_p, theta_p, x, x_int, 1e-10, 1e-10, 1e8);
  y_hat = integrate_ode_bdf(sho, y0_p, t0_p, ts_p, theta_d, x, x_int, 1e-10, 1e-10, 1e8);
  y_hat = integrate_ode_bdf(sho, y0_p, t0_p, ts_p, theta_p, x, x_int, 1e-10, 1e-10, 1e8);
}
