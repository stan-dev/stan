functions {
  real[] harm_osc_ode(real[] t,
                      real[] y,         // state
                      real[] theta,     // parameters
                      real[] x,         // data
                      int[] x_int) {    // integer data
    real dydt[2];
    dydt[1] <- x[1] * y[2];
    dydt[2] <- -y[1] - theta[1] * y[2];
    return dydt;
  }
}
data {
  real y0[2];
  real t0;
  real ts[10];
  real x[1];   
  int x_int[0];
  real y[10,2];
}
parameters {
  real theta[1];
  real<lower=0> sigma;
}
transformed parameters {
  real y_hat[10,2];
  y_hat <- integrate_ode_adams(harm_osc_ode,  // system
                     y0,            // initial state
                     t0,            // initial time
                     ts,            // solution times
                     theta,         // parameters
                     x,             // data
                     x_int, 0.01, 0.01, 10);        // integer data
  
}
model {
  for (t in 1:10)
    y[t] ~ normal(y_hat[t], sigma);  // independent normal noise
}
