functions {
  real my_fun3(real x);

  real my_fun3(data real x) {
    return 2 * x;
  }
  
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
