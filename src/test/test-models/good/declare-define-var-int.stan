functions {
  int foo() {
    int lf1 = 3;
    print("foo ",lf1);
    return lf1;
  }
}
transformed data {
  int td1 = 1;
  int td_a1[td1];
  int td_a2[td1] = td_a1;
  print("transformed data ",td_a2);
}
transformed parameters {
  real p1;
  {
    int lp1 = 1;
    print("transformed param ",lp1);
  }
}
model {
  // local variables
  int lm1 = 4;
  print("local int ",lm1);
  print(foo());
}
generated quantities {
  int gq1 = 1;
  print("gq1 ",gq1);
  gq1 = 2;
  {
    int lgq1 = 2;
    print("gq2 ",lgq1);
    lgq1 = 2;
  }
}
