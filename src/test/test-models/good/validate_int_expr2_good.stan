parameters {
  vector[10] y;
}
model {
  for (n in 1:10)
    y ~ normal(0,1);
}
