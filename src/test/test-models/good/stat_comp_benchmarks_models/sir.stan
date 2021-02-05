functions {
  array[] real simple_SIR(real t, array[] real y, array[] real theta,
                          array[] real x_r, array[] int x_i) {
    array[4] real dydt;
    dydt[1] = -theta[1] * y[4] / (y[4] + theta[2]) * y[1];
    dydt[2] = theta[1] * y[4] / (y[4] + theta[2]) * y[1] - theta[3] * y[2];
    dydt[3] = theta[3] * y[2];
    dydt[4] = theta[4] * y[2] - theta[5] * y[4];
    return dydt;
  }
}
data {
  int<lower=0> N_t;
  array[N_t] real t;
  array[4] real y0;
  array[N_t] int stoi_hat;
  array[N_t] real B_hat;
}
transformed data {
  real t0 = 0;
  real<lower=0> kappa = 1000000;
  array[0] real x_r;
  array[0] int x_i;
}
parameters {
  real<lower=0> beta;
  real<lower=0> gamma;
  real<lower=0> xi;
  real<lower=0> delta;
}
transformed parameters {
  array[N_t, 4] real<lower=0> y;
  {
    array[5] real theta = {beta, kappa, gamma, xi, delta};
    y = integrate_ode_rk45(simple_SIR, y0, t0, t, theta, x_r, x_i);
  }
}
model {
  beta ~ cauchy(0, 2.5);
  gamma ~ cauchy(0, 1);
  xi ~ cauchy(0, 25);
  delta ~ cauchy(0, 1);
  stoi_hat[1] ~ poisson(y0[1] - y[1, 1]);
  for (n in 2 : N_t) 
    stoi_hat[n] ~ poisson(y[n - 1, 1] - y[n, 1]);
  B_hat ~ lognormal(log(col(to_matrix(y), 4)), 0.15);
}

