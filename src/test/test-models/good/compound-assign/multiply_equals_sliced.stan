functions {
  void foo_vec(real a1) {
    int J[2];
    matrix[2, 2] aa;
    matrix[3, 4] bb;
    bb[1:2,1:2] *= aa;  // matrix *= matrix
    bb[1:2,1:2] *= a1;  // matrix *= real
    aa[1, J] *= bb[1:2,1:2];  // row_vector *= matrix
    aa[1, J] *= a1;  // row_vector *= real
    aa[J, 1] *= a1;  // vector *= real
  }
}
