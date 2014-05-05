functions {
  real barfoo(real x);
  // error not defining barfoo
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
