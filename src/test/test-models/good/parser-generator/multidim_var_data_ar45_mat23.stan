data {
  array[4, 5] matrix<lower=0, upper=1>[2, 3] ar_mat;
}
transformed data {
  for (i in 1 : 4) {
    for (j in 1 : 5) {
      matrix[2, 3] foo = ar_mat[i, j];
      print("ar dim1: ", i, " ar dim2: ", j, " matrix: ", foo);
    }
  }
}

