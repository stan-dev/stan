parameters {
  vector[10.9] y;
}
model {
  for (n in 1:10.3)
    y ~ normal(0,1);
}
