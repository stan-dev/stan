functions {
  void foo_vec(int a1) {
    int J[a1];
    matrix[2, 2] aa;
    matrix[3, 4] bb;
    aa[J,1] += bb[1:2,1:2];
  }
}
