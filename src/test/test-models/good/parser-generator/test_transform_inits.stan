parameters {
  real<lower=0> nu;
  real<lower=0, upper=(nu == 0 ? 1 : positive_infinity())> lambda;
}

