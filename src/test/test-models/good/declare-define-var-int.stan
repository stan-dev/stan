functions {
  int foo() {
    int lf1 = 1;
    return lf1;
  }
}
transformed data {
  int td1 = 1;
  int td_a1[5];
  int td_a2[5] = td_a1;
}
transformed parameters {
  real p1;
  {
    int lp1 = 1;
  }
}
model {
  // local variables
  int lm1 = 1;
}
generated quantities {
  int gq1 = 1;
  gq1 = 2;
  {
    int lgq1 = 1;
    lgq1 = 2;
  }
}
