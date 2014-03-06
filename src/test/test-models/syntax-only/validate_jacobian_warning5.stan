parameters {
  vector[10] y;
}
model {
  head((y .* y), 2)  ~ normal(0,1);
}
