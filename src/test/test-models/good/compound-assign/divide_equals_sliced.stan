functions {
  void foo_vec(real a1) {
    array[2] int J;
    matrix[2, 2] aa;
    matrix[3, 4] bb;
    bb[1 : 2, 1 : 2] /= a1;
    aa[1, J] /= a1;
    aa[J, 1] /= a1;
  }
}

