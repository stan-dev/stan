functions {
  real[] ode(real t, real[] y, real[] theta, real[] x, int[] x_int) {
    real dydt[0];
    return dydt;
  }
}
data {
  int<lower=1> T;
  real t0;
  real y0[0];
  real ts[T];
  real y[T,2];
}
transformed data {
  real x[0];
  int x_int[0];
}
parameters {
  real theta[0];
}
transformed parameters {
  real y_hat[T,2]; 
  {
    int N = 0;
    y_hat = integrate_ode(ode, y0, t0, segment(ts, 0, N), theta, x, x_int);
  }
}
model {
}
