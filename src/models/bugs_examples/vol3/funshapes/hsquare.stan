/*
 * Hollow Square
 * http://www.openbugs.info/Examples/Funshapes.html
 *
 * Transform a uniform rectangle
 *    1234
 *    5678
 *    90AB
 * to a uniform hollow square
 *    1234
 *    5  6
 *    7  8
 *    90AB
 */
parameters {
  real<lower=0,upper=2> x_raw; 
  real<lower=0,upper=1.5> y_raw; 
} 
model {
  /* no-op; uniformity implied by parameter constraints */
} 
generated quantities {
  real x;
  real y;
  if (y_raw > 1) {
    // cases 1, 2, 3, 4
    x <- x_raw - 1;
    y <- y_raw - 0.5;
  } else if (y_raw < 0.5) {
    // cases 9, 0, A, B
    x <- x_raw - 1;
    y <- y_raw - 1;
  } else if (x_raw < 0.5) {
    // case 5
    x <- x_raw - 1;
    y <- y_raw - 0.5;
  } else if (x_raw < 1.0) {
    // case 6
    x <- x_raw - 1.5;
    y <- y_raw - 1;
  } else if (x_raw < 1.5) {
    // case 7
    x <- x_raw - 0.5;
    y <- y_raw - 0.5;
  } else {
    x <- x_raw - 1.0;
    y <- y_raw - 1.0;
  }
}
