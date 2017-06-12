functions {
  void foo(real x) {
    print("quitting time");
    reject("user-specified rejection");
  }
}
transformed data {
  print("In transformed data");
}
parameters {
  real y;
} 
transformed parameters {
  print("In transformed parameters");
}
model {
  real v;
  print("In model block.");
  y ~ normal(0,1);
  foo(v)
}
generated quantities {
  print("In generated quantities");
}
