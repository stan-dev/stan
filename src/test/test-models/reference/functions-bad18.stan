functions {
  void foo() {
    print("hello, world!");
  }
}
parameters {
  real foo;  // should be name conflict here
}
model {
  foo ~ normal(0.0, 1.0);
  foo();
}
