transformed data {
  int a1;
  real d1;
  {
    int a2;
    real d2;
    array[20, 30] real b;
    array[60, 70] matrix[40, 50] ar_mat;
    ar_mat[1, 1, 1, 1] = 1.0;
    a1 = a2;
    d1 = b[1, 1];
  }
}

