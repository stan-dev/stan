// test via printf, compile and then run (cmdstan):
// > stan/src/test/test-models/good/compound-assign/plus_equals_prim_print sample algorithm=fixed_param num_warmup=0 num_samples=1

functions {
  real foo(real a1) {
    real b = a1;
    print(" in function foo");
    print("    b: ", b);
    print("    a1: ", a1);
    b += a1 / 2;
    print("    b += a1: ", b);
    return b;
  }
}
transformed data {
  int x = 10;
  real y = 20;
  print("in transformed data");
  print("x: ", x)
  x += 1;  // scalar int
  print("x += 1: ", x)
  print("y: ", y);
  y += 1;  // scalar double
  print("y += 1: ", y);
}
transformed parameters {
  real w = 30;
  print("in transformed parameters");
  print("w: ", w);
  w += y;
  print("w += y: ", w);
  w += foo(w);
  print("w += foo(w): ", w);
}  
model {
  real v = 7;
  print("in model block");
  v += y;
  print("v += y: ", v);
  v += foo(w);
  print("v += foo(w): ", v);
  v += foo(y);
  print("v += foo(y): ", v);
}
generated quantities {
  real z = 40;
  print("in generated quantities");
  print("z: ", z);
  print("y: ", y);
  print("w: ", w);
  z += w;
  print("z += w: ", z);
  z += y;
  print("z += y: ", z);
  z += foo(y);
  print("z += foo(y): ", z);
}
