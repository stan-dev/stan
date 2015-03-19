functions {
  real relative_diff(real x, real y, real max, real min) {
    real abs_diff;
    real avg_scale;
    abs_diff <- fabs(x - y);
    avg_scale <- (fabs(x) + fabs(y)) / 2;
    if ((abs_diff / avg_scale) > max)
      reject("user-specified rejection, difference above ",max," x:",x," y:",y);
    if ((abs_diff / avg_scale) < min)
      reject("user-specified rejection, difference below ",min," x:",x," y:",y);
    return abs_diff / avg_scale;
  }    
}
transformed data {
  real a;
  real b;
  real mx;
  real mn;
  a <- -9.0;
  b <- -1.0;
  mx <- 1.2;
  mn <- 1.1;
}
parameters {
  real y;
}
model {
  real c;
  c <- relative_diff(a,b,mx,mn);
  y ~ normal(0,1);
}
