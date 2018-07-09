functions {
  void foo_vec(real a1) {
    int J[2];
    matrix[2, 2] aa;
    matrix[3, 4] bb;
    row_vector[2] cc;
    vector[2] dd;
    bb[1:2,1:2] ./= aa;  // matrix, matrix
    aa[1, J] ./= cc;  // row_vector, row_vector
    aa[J, 1] ./= dd;  // vector, vector
  }
}
