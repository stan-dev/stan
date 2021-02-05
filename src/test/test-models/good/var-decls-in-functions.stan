functions {
  array[] real harm_osc_ode(real t, array[] real y, array[] real theta,
                            array[] real x, array[] int x_int) {
    array[size(y)] real dydt;
    return dydt;
  }
}
parameters {
  real y;
}
model {
  y ~ normal(0, 1);
}

