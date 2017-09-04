functions {
  real[] sho(real t,
             real[] y,
             real[] theta,
             real[] x_r,
             int[] x_i) {
    real dydt[2];
    dydt[1] = y[2];
    dydt[2] = -y[1] - theta[1] * y[2];
    return dydt;
  }

  real[,] do_integration_nested(real[] y0, real t0, real[] ts, real[] theta, data matrix xmat_r) {
    int x_i[0];
    return(integrate_ode_rk45(sho, y0, t0, ts, theta, to_array_1d(xmat_r[1]), x_i));
  }

  real[,] do_integration(real[] y0, real t0, real[] ts, real[] theta, matrix xmat_r, real[] x_r) {
    matrix[2,2] xmat_sub_r;
    xmat_sub_r <- block(xmat_r, 1, 1, 2, 2);
    return(do_integration_nested(y0, t0, ts, theta, xmat_sub_r));
  }
}
data {
  int<lower=1> T;
  real y[T,2];
  real t0;
  real ts[T];
}
transformed data {
  real x_r[1];
  matrix[2,2] xmat_r;
}
parameters {
  real y0[2];
  vector<lower=0>[2] sigma;
  real theta[1];
}
transformed parameters {
  real y_hat[T,2];
  y_hat = do_integration(y0, t0, ts, theta, xmat_r, x_r);
}
model {
  sigma ~ cauchy(0, 2.5);
  theta ~ normal(0, 1);
  y0 ~ normal(0, 1);
  for (t in 1:T)
    y[t] ~ normal(y_hat[t], sigma);
}
