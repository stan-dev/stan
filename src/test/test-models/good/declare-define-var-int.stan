transformed data {
  int td1 = 1;     // int_d <- int_d
  int tdIntArray1[7];
  int tdIntArray2[7] = tdIntArray1;
}
transformed parameters {
  real p1;
  p1 = 1;
  // local variables
  {
    int lp1 = 1;      // real_p <- int_d 
  }
}
model {
  // local variables
  int lm1 = 1;      // int_d <- int_d 
}
generated quantities {
  int gq1 = 1;     // int_d <- int_d
  gq1 = 2;
  {
    int lgq1 = 1;     // int_d <- int_d
    lgq1 = 2;
  }
}
