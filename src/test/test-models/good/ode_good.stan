functions {
  array[] real harm_osc_ode(real t, array[] real y, array[] real theta,
                            array[] real x, array[] int x_int) {
    array[2] real dydt;
    dydt[1] = x[1] * y[2];
    dydt[2] = -y[1] - theta[1] * y[2];
    return dydt;
  }
}
data {
  array[2] real y0;
  real t0;
  array[10] real ts;
  array[1] real x;
  array[0] int x_int;
  array[10, 2] real y;
}
parameters {
  array[1] real theta;
  real<lower=0> sigma;
}
transformed parameters {
  array[10, 2] real y_hat;
  y_hat = integrate_ode(harm_osc_ode, y0, t0, ts, theta, x, x_int);
  y_hat = integrate_ode_rk45(harm_osc_ode, y0, t0, ts, theta, x, x_int);
  y_hat = integrate_ode_bdf(harm_osc_ode, y0, t0, ts, theta, x, x_int);
  y_hat = integrate_ode_adams(harm_osc_ode, y0, t0, ts, theta, x, x_int);
  y_hat = integrate_ode_rk45(harm_osc_ode, y0, t0, ts, theta, x, x_int, 0.01,
                             0.01, 10);
  y_hat = integrate_ode_bdf(harm_osc_ode, y0, t0, ts, theta, x, x_int, 0.01,
                            0.01, 10);
  y_hat = integrate_ode_adams(harm_osc_ode, y0, t0, ts, theta, x, x_int,
                              0.01, 0.01, 10);
}
model {
  for (t in 1 : 10) 
    y[t] ~ normal(y_hat[t], sigma);
}

