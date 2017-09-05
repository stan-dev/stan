functions {
  real my_fun3(data real x);

  real my_fun3(data real x) {
    return 2 * x;
  }
  
}
parameters {
  real y;
}
model {
  real z = my_fun3(y);
  y ~ normal(0,1);
}
