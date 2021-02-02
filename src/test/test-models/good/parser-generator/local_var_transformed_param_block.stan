transformed parameters {
  real p2;
  for (x in 1 : 10) 
    p2 = x;
  {
    int a;
    array[2] int vs;
    array[20, 30] real b;
    array[60, 70] matrix[40, 50] ar_mat;
    ar_mat[1, 1, 1, 1] = 1.0;
    p2 = b[1, 1];
  }
}

