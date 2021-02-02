functions {
  vector foo(vector x, real y) {
    if (y < 10) 
      return log(x);
    else 
      return log(y * x);
  }
}
transformed data {
  vector[3] x;
  x = rep_vector(0, 3);
}
parameters {
  real y;
}
model {
  foo(x, y) ~ normal(0, 1);
}

