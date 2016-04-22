transformed data {
  real a;
  a <- inc_beta(1, 1, 1);
  a <- inc_beta(1, 1, 2.7);
  a <- inc_beta(1, 2.7, 1);
  a <- inc_beta(1, 2.7, 2.7);
  a <- inc_beta(2.7, 1, 1);
  a <- inc_beta(2.7, 1, 2.7);
  a <- inc_beta(2.7, 2.7, 1);
  a <- inc_beta(2.7, 2.7, 2.7);
}  
parameters {
  real b;
}
transformed parameters {
  real c;
  c <- inc_beta(b, b, b);
  c <- inc_beta(b, b, 2.7);
  c <- inc_beta(b, 2.7, b);
  c <- inc_beta(b, 2.7, 2.7);
  c <- inc_beta(2.7, b, b);
  c <- inc_beta(2.7, b, 2.7);
  c <- inc_beta(2.7, 2.7, b);
}
model {
  b ~ normal(0, 1);
}
  
