parameters {
  real y;
}
model {
  y ~ normal(0,1);

  for (i in 1:10) {
    real x[i];  // should allow i here.
    for (j in 1:i)
      x[j] <- j * j;
  }
}
