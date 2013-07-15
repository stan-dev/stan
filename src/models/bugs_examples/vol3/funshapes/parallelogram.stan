/*
 * BUGS Volume 3, funshapes, parallelogram
 * http://www.openbugs.info/Examples/Funshapes.html
 */
parameters {
  real<lower=0,upper=1> x; 
  real<lower=0,upper=1> y_raw; 
} 
transformed parameters {
  real<lower=-1,upper=1> y;
  y <- y_raw - x;
}
model {
  /* uniform within constraints */
} 
