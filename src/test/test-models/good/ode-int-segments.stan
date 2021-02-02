functions {
  array[] real ode(real t, array[] real y, array[] real theta,
                   array[] real x, array[] int x_int) {
    array[0] real dydt;
    return dydt;
  }
}
data {
  int<lower=1> T;
  real t0;
  array[0] real y0;
  array[T] real ts;
  array[T, 2] real y;
}
transformed data {
  array[0] real x;
  array[0] int x_int;
}
parameters {
  array[0] real theta;
}
transformed parameters {
  array[T, 2] real y_hat;
  {
    int N = 0;
    y_hat = integrate_ode(ode, y0, t0, segment(ts, 0, N), theta, x, x_int);
  }
}
model {

}

