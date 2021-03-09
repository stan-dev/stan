functions {
  array[] real one_comp_mm_elim_abs(real t, array[] real y,
                                    array[] real theta, array[] real x_r,
                                    array[] int x_i) {
    array[1] real dydt;
    real k_a = theta[1];
    real K_m = theta[2];
    real V_m = theta[3];
    real D = x_r[1];
    real V = x_r[2];
    real dose = 0;
    real elim = (V_m / V) * y[1] / (K_m + y[1]);
    if (t > 0) 
      dose = exp(-k_a * t) * D * k_a / V;
    dydt[1] = dose - elim;
    return dydt;
  }
}
data {
  real t0;
  array[1] real C0;
  real D;
  real V;
  int<lower=1> N_t;
  array[N_t] real times;
  array[N_t] real C_hat;
}
transformed data {
  array[2] real x_r = {D, V};
  array[0] int x_i;
}
parameters {
  real<lower=0> k_a;
  real<lower=0> K_m;
  real<lower=0> V_m;
  real<lower=0> sigma;
}
transformed parameters {
  array[N_t, 1] real C;
  {
    array[3] real theta = {k_a, K_m, V_m};
    C = integrate_ode_bdf(one_comp_mm_elim_abs, C0, t0, times, theta, x_r,
                          x_i);
  }
}
model {
  k_a ~ cauchy(0, 1);
  K_m ~ cauchy(0, 1);
  V_m ~ cauchy(0, 1);
  sigma ~ cauchy(0, 1);
  for (n in 1 : N_t) 
    C_hat[n] ~ lognormal(log(C[n, 1]), sigma);
}
generated quantities {
  array[N_t] real C_ppc;
  for (n in 1 : N_t) 
    C_ppc[n] = lognormal_rng(log(C[n, 1]), sigma);
}

