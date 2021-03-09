parameters {
  array[2] real<lower=-10, upper=10> y;
}
model {
  y ~ normal(0, 1);
}
generated quantities {
  print("no QoIs");
}

