functions {
  int foo() {
    int lf1 = 3;
    print("foo ", lf1);
    return lf1;
  }
}
data {
  int n;
  array[n, n] int d;
}
transformed data {
  int td1 = 1;
  int td2 = td1;
  array[n] int td_a1 = d[1];
  array[n] int td_a2 = td_a1;
  array[n, n] int td_a3 = d;
  array[n] int td_a4 = td_a3[n];
  int td3 = td_a3[2, 2];
  print("td1 = ", td1);
  print("td2 = ", td2);
  print("td3 = ", td3);
  print("td_a3 = ", td_a3);
  print("transformed data td2 ", td2);
  print("transformed data td_a2 ", td_a2);
  print("transformed data td_a4 ", td_a4);
}
transformed parameters {
  real p1;
  {
    int lp1 = 1;
    print("transformed param ", lp1);
  }
}
model {
  int lm1 = 4;
  print("local int ", lm1);
  print(foo());
}
generated quantities {
  int gq1 = 1;
  print("gq1 ", gq1);
  gq1 = 2;
  {
    int lgq1 = 2;
    print("gq2 ", lgq1);
    lgq1 = 2;
  }
}

