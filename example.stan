parameters {
  real mu1;
  real mu2;
}

model {
  mu1 ~ normal(0, 10);
  mu2 ~ normal(0, 1);
}
