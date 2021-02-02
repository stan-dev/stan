functions {
  void bar(array[,] matrix a) {
    print("hello world");
  }
}
data {
  array[4, 5] matrix<lower=0, upper=1>[2, 3] ar_mat;
}
transformed data {
  int td_a = 1;
  real td_b = 2;
  array[4, 5] matrix<lower=0, upper=1>[2, 3] td_ar_mat;
  for (i in 1 : 4) {
    for (j in 1 : 5) {
      matrix[2, 3] foo = ar_mat[i, j];
      print("ar dim1: ", i, " ar dim2: ", j, " matrix: ", foo);
    }
  }
  bar(td_ar_mat);
}
parameters {
  real p_b;
  array[4, 5] matrix<lower=0, upper=1>[2, 3] p_ar_mat;
}
transformed parameters {
  real tp_b = 2;
  array[4, 5] matrix[2, 3] tp_ar_mat = ar_mat;
}
generated quantities {
  real gq_b = 2;
  array[4, 5] matrix[2, 3] gq_ar_mat = ar_mat;
}

