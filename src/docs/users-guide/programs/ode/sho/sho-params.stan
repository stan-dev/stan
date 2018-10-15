functions {
  real[] sho(real t,
             real[] y, 
             real[] theta,
             real[] x,
             int[] x_int) {
    real dydt[2];
    dydt[1] <- y[2];
    dydt[2] <- -y[1] - theta[1] * y[2];
    return dydt;
  }
}
data {
  int<lower=1> T;
  real y[T,2];
  real y0[2];
  real t0;
  real ts[T];
}
transformed data {
  real x[0];
  int x_int[0];
}
parameters {
  real theta[1];
  vector<lower=0>[2] sigma;
}
model {
  real y_hat[T,2];
  sigma ~ cauchy(0,2.5);
  y_hat <- integrate_ode(sho, y0, t0, ts, theta, x, x_int);
  for (t in 1:T)
    y[t] ~ normal(y_hat[t], sigma);
}
