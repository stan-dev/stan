parameters {
  vector[5] v;
}
model {
  (v' * v) ~ normal(0,1);
}
