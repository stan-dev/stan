
transformed data {
  real<offset=2> a = 3;
}
parameters {
  matrix<offset=-412,multiplier=3>[3,1] theta[2];
  real<offset=1,multiplier=5> x;
  real<offset=42> w;
  vector<multiplier=242>[3] ww;
  row_vector<multiplier=242>[3] www;
}
transformed parameters {
  real<offset=23> y = x^2;
}
model {
  y ~ normal(0,1);
}
generated quantities {
  real<offset=1> z;
  z=3;
}