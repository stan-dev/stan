parameters {
  real y;
}
model {
  y ~ normal(0,1);
  raise_exception("user-specified exception");
}
generated quantities {
  print("generating quantities");
}

