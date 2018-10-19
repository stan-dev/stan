
transformed data {
  real<location=2> a = 3;
}
parameters {
  matrix<location=-412,scale=3>[3,1] theta[2];
  real<location=1,scale=5> x;
  real<location=42> w;
  vector<scale=242>[3] ww;
  row_vector<scale=242>[3] www;
}
transformed parameters {
  real<location=23> y = x^2;
}
model {
  y ~ normal(0,1);
}
generated quantities {
  real<location=1> z;
  z=3;
}