functions {
  int foo(int a);
  int foo(int a) {
    return a;
  }
  int bar(int a) {
    return a;
  }
}
parameters {
  real theta;
}
model {
  theta ~ normal(0, 1);
}
