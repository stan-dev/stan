functions {
  vector foo(vector a1) {
    vector[3] lf1;
    lf1 = a1;
    print(" in function foo");
    print("    lf1: ",lf1);
    print("    a1: ",a1);
    lf1 += a1;
    print("    lf1 += a1: ",lf1);
    return lf1;
  }
}
generated quantities {
  vector[3] z = [1, 2, 3]';
  vector[4] ident = [1, 1, 1, 1]';
  print("in generated quantities");
  print("z: ", z);
  z += ident;
  print("z += ident ", z);
  z += foo(ident);
  print("z += foo(ident) ", z);
  z[1] += 5;
  print("z[1] += 5 ", z);
}
