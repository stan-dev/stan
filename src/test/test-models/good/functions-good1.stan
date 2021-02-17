functions {
  real foo0() {
    return 0.0;
  }
  real foo1(real x) {
    return 1.0;
  }
  real foo2(real x, real y) {
    return 2.0;
  }
}
data {
  array[6] int<lower=0> N;
}
transformed data {
  real a;
  real b;
  real c;
  a = foo0();
}

