functions {
  void foo(real x) {
    print("quitting time");
    reject("user-specified rejection");
  }
}
transformed data {
  int x;
  print("In transformed data");
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
  real v;
  print("In generated quantities");
  foo(v)
}
