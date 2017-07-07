generated quantities {
  real x = 3.3;
  matrix[2, 2] z = [[1, 2], [4, 5]];
  matrix[2, 2] ident = [[1, 0], [0, 1]];
  print("z ", z);
  print("x ", x);
  print("ident ", ident);
  z *= x;
  print("z *= x ", z);

  z = [[1, 2], [4, 5]];
  z += x;
  print("z += x ", z);

  z = [[1, 2], [4, 5]];
  z -= x;
  print("z -= x ", z);

  z = [[1, 2], [4, 5]];
  z /= x;
  print("z /= x ", z);

  z = [[1, 2], [4, 5]];
  z *= ident;
  print("z *= ident ", z);

  z = [[1, 2], [4, 5]];
  z .*= ident;
  print("z .*= ident ", z);

  z = [[1, 2], [4, 5]];
  z += ident;
  print("z += ident ", z);

  z = [[1, 2], [4, 5]];
  z -= ident;
  print("z -= ident ", z);

  z = [[1, 2], [4, 5]];
  z ./= ident;
  print("z ./= ident ", z);
}
