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
}
model {
  real y_hat[T,2];
  y_hat = integrate_ode_rk45(sho, y0_d, t0, ts, theta_d, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_d, t0, ts, theta_p, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_p, t0, ts, theta_d, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_p, t0, ts, theta_p, x, x_int);
}
generated quantities {
  real y_hat[T,2];
  y_hat = integrate_ode_rk45(sho, y0_d, t0, ts, theta_d, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_d, t0, ts, theta_p, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_p, t0, ts, theta_d, x, x_int);
  y_hat = integrate_ode_rk45(sho, y0_p, t0, ts, theta_p, x, x_int);
}
