// compiles, should throw run-time error, mismatched sizes
generated quantities {
  vector[3] z = [1, 2, 3]';
  vector[4] ident = [1, 1, 1, 1]';
  z += ident;
}
