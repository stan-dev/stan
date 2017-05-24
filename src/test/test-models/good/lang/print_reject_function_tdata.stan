functions {
  void foo(real x) {
    print("quitting time");
    reject("user-specified rejection");
  }
}
transformed data {
  real v;
  print("In transformed data");
  foo(v)
}
parameters {
  real y;
} 
transformed parameters {
  print("In transformed parameters");
}
model {
  print("In model block.");
  y ~ normal(0,1);
}
generated quantities {
  print("In generated quantities");
}
