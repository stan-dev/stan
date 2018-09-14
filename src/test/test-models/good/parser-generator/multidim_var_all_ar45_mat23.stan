functions {
  void bar(matrix[,] a) {
    print("hello world");
  }
}
data {
  matrix<lower=0,upper=1>[2,3] ar_mat[4,5];
}
transformed data {
  int td_a = 1;
  real td_b = 2;
  matrix<lower=0,upper=1>[2,3] td_ar_mat[4,5];
  for (i in 1:4) {
    for (j in 1:5) {
      matrix[2,3] foo = ar_mat[i,j];
      print("ar dim1: ",i, " ar dim2: ",j, " matrix: ", foo);
    }
  }
  bar(td_ar_mat);
}
parameters {
  real p_b;
  matrix<lower=0,upper=1>[2,3] p_ar_mat[4,5];
}
transformed parameters {
  real tp_b = 2;
  matrix[2,3] tp_ar_mat[4,5] = ar_mat;
}
generated quantities {
  real gq_b = 2;
  matrix[2,3] gq_ar_mat[4,5] = ar_mat;
}
