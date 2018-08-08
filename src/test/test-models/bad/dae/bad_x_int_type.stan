functions {
  real[] prey_predator_harvest(real t,
                               real[] yy, // state
                               real[] yp, // state derivative
                               real[] theta,  // parameters
                               real[] x,      // data
                               int[] x_int) {
    real res[3];
    real r1;
    real r2;
    real p;
    r1 = 1.0;
    r2 = 3.0;
    p = 2.0;

    res[1] = yp[1] - yy[1] * (r1 - yy[2]);
    res[2] = yp[2] - yy[2] * (r2 - yy[2] / yy[1] - yy[3]);
    res[3] = yy[3] * (p * yy[2] - 1.0) - theta[1];

    return res;
  }
}

data {
  real yy0[5];
  real yp0[3];
  real t[2];
  real ts[10];
  real x[1,3];   
  int x_int_bad[3,1];
  real y[10,3];
}
parameters {
  real theta[2];
  real<lower=0> sigma;
}
transformed parameters {
  real y_hat[10,3];
  y_hat = integrate_dae(prey_predator_harvest, // system
                        yy0, // initial state
                        yp0,            // initial derivative
                        t,             // initial time
                        ts,             // solution times
                        theta,          // parameters
                        x,              // data
                        x_int_bad,          // integer data
                        1.E-3, 1.E-10, 1000);
}
model {
  for (t in 1:10)
    y[t] ~ normal(y_hat[t], sigma);  // independent normal noise
}
