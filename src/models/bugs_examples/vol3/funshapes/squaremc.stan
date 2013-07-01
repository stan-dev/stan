/*
 * BUGS Volume 3, funshapes, square minus circle
 * http://www.openbugs.info/Examples/Funshapes.html
 *
 * first draw raw samples from diamond-like shape reflected per
 * quadrant, then reflect in transformed parameters
 *
 * unfortunate redundancy in 1-sqrt() term because we can't
 * get local variables into parameters
 */
parameters {
  real<lower=-1,upper=1> x_raw;
  real<lower = -(1 - sqrt(1 - square(1 - fabs(x_raw)))),
       upper =  (1 - sqrt(1 - square(1 - fabs(x_raw))))> y_raw;
}
transformed parameters {
  real<lower=-1,upper=1> x;
  real<lower=-1,upper=1> y;
  x <- if_else(x_raw > 0, 1, -1) - x_raw;
  y <- if_else(y_raw > 0, 1, -1) - y_raw;
}
model {
  lp__ <- lp__ + log1m(sqrt(1 - square(1 - fabs(x_raw))));
}
