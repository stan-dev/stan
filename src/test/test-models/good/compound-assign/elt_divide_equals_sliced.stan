functions {
  void foo_vec(real a1) {
    array[2] int J;
    matrix[2, 2] aa;
    matrix[3, 4] bb;
    row_vector[2] cc;
    vector[2] dd;
    bb[1 : 2, 1 : 2] ./= aa;
    aa[1, J] ./= cc;
    aa[J, 1] ./= dd;
  }
}

