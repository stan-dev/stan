functions {
  real[] harm_osc_ode(real t,           // time
                      real[] y,         // state
                      real[] theta,     // parameters
                      real[] x,         // data
                      int[] x_int) {    // integer data
    real dydt[size(y)];
    // ... set dydt at state y and time t ...
    return dydt;
  }
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
