// compiles, should throw run-time error, mismatched sizes
generated quantities {
  row_vector[3] z = [1, 2, 3];
  row_vector[4] ident = [1, 1, 1, 1];
  z += ident;
}
