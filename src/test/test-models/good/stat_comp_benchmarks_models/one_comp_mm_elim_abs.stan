functions {
  real[] one_comp_mm_elim_abs(real t,
                              real[] y,
                              real[] theta,
                              real[] x_r,
                              int[] x_i) {
    real dydt[1];
    real k_a = theta[1]; // Dosing rate in 1/day
    real K_m = theta[2]; // Michaelis-Menten constant in mg/L
    real V_m = theta[3]; // Maximum elimination rate in 1/day
    real D = x_r[1];
    real V = x_r[2];
    real dose = 0;
    real elim = (V_m / V) * y[1] / (K_m + y[1]);

    if (t > 0)
      dose = exp(- k_a * t) * D * k_a / V;

    dydt[1] = dose - elim;

    return dydt;
  }
}

data {
  real t0;    // Initial time in days;
  real C0[1]; // Initial concentration at t0 in mg/L

  real D;   // Total dosage in mg
  real V;   // Compartment volume in L

  int<lower=1> N_t;
  real times[N_t];   // Measurement times in days

  // Measured concentrations in effect compartment in mg/L
  real C_hat[N_t];
}

transformed data {
  real x_r[2] = {D, V};
  int x_i[0];
}

parameters {
  real<lower=0> k_a; // Dosing rate in 1/day
  real<lower=0> K_m; // Michaelis-Menten constant in mg/L
  real<lower=0> V_m; // Maximum elimination rate in 1/day
  real<lower=0> sigma;
}

transformed parameters {
  real C[N_t, 1];
  {
    real theta[3] = {k_a, K_m, V_m};
    C = integrate_ode_bdf(one_comp_mm_elim_abs, C0, t0, times, theta, x_r, x_i);
  }
}

model {
  // Priors
  k_a ~ cauchy(0, 1);
  K_m ~ cauchy(0, 1);
  V_m ~ cauchy(0, 1);
  sigma ~ cauchy(0, 1);

  // Likelihood
  for (n in 1:N_t)
    C_hat[n] ~ lognormal(log(C[n, 1]), sigma);
}

generated quantities {
  real C_ppc[N_t];
  for (n in 1:N_t)
    C_ppc[n] = lognormal_rng(log(C[n, 1]), sigma);
}
