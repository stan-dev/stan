functions {
  real foo(real x) {
    return x + get_lp();
  }
}
parameters {
  real z;
}
model {
  z ~ normal(0,1);
  print(foo(z));
}
