transformed data {
  real y[10,10];
  real y0;
  real t0;
  real ts[10];
  real theta[3];
  real x[2];
  int x_int[4];
  y <- integrate_ode(foo,y0,t0,ts,theta,x x_int);
}
model {
}
