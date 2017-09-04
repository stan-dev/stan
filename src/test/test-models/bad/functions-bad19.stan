functions {
  real my_fun3(data real x);

  real my_fun3(real x) {
    return 2 * x;
  }
  
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
