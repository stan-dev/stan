/**
 * get_lp() allowed in:
 *  - functions ending in _lp
 *  transformed parameter block
 *  model block
 */
functions {
  // allowed in functions ending in _lp
  real foo_lp(real x) {
    return x + get_lp();
  }
}
parameters {
  real y;
}
transformed parameters {
  real z;
  z <- get_lp();  
}
model {
  real w;
  w <- get_lp();
  y ~ normal(0,1);
}
