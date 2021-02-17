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
transformed data {
  int N_t = 20;
  array[N_t] real times;
  real t0 = 0;
  array[1] real C0 = {0.0};
  array[3] real theta = {0.75, 0.25, 1};
  real sigma = 0.1;
  array[2] real x_r = {30.0, 2.0};
  array[0] int x_i;
  for (n in 1 : N_t) 
    times[n] = 0.5 * n;
}
model {

}
generated quantities {
  real t_init = t0;
  array[1] real C_init = {C0[1]};
  real D = x_r[1];
  real V = x_r[2];
  array[N_t] real ts;
  array[N_t, 1] real C;
  array[N_t] real C_hat;
  for (n in 1 : N_t) 
    ts[n] = times[n];
  C = integrate_ode_bdf(one_comp_mm_elim_abs, C0, t0, times, theta, x_r, x_i);
  for (n in 1 : N_t) 
    C_hat[n] = lognormal_rng(log(C[n, 1]), sigma);
}

