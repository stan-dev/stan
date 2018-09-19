// Simple SIR model inspired by the presentation in
// http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3380087/pdf/nihms372789.pdf

functions {
  
  // theta[1] = beta, water contact rate
  // theta[2] = kappa, C_{50}
  // theta[3] = gamma, recovery rate
  // theta[4] = xi, bacteria production rate
  // theta[5] = delta, bacteria removal rate
  real[] simple_SIR(real t,
                    real[] y,
                    real[] theta,
                    real[] x_r,
                    int[] x_i) {

    real dydt[4];

    dydt[1] = - theta[1] * y[4] / (y[4] + theta[2]) * y[1];
    dydt[2] = theta[1] * y[4] / (y[4] + theta[2]) * y[1] - theta[3] * y[2];
    dydt[3] = theta[3] * y[2];
    dydt[4] = theta[4] * y[2] - theta[5] * y[4];

    return dydt;
  }
}

data {
  int<lower=0> N_t;
  real t[N_t];
  real y0[4];
  int stoi_hat[N_t];
  real B_hat[N_t];
}

transformed data {
  real t0 = 0;
  real<lower=0> kappa = 1000000;

  real x_r[0];
  int x_i[0];
}

parameters {
  real<lower=0> beta;
  real<lower=0> gamma;
  real<lower=0> xi;
  real<lower=0> delta;
}

transformed parameters {
  real<lower=0> y[N_t, 4];
  {
    real theta[5] = {beta, kappa, gamma, xi, delta};
    y = integrate_ode_rk45(simple_SIR, y0, t0, t, theta, x_r, x_i);
  }
}
  
model {
  beta ~ cauchy(0, 2.5);
  gamma ~ cauchy(0, 1);
  xi ~ cauchy(0, 25);
  delta ~ cauchy(0, 1);

  stoi_hat[1] ~ poisson(y0[1] - y[1, 1]);
  for (n in 2:N_t)
    stoi_hat[n] ~ poisson(y[n - 1, 1] - y[n, 1]);

  B_hat ~ lognormal(log(col(to_matrix(y), 4)), 0.15);
  
}
