functions {
  real foo() {
    real lf1 = 1;     // real_d <- int_d
    real lf2 = 2.0;   // real_d <- real_d
    real lf3[5];
    real lf4[5] = lf3;
    lf1 = lf3[1] + lf4[1];
    return lf1;
  }
}
transformed data {
  real td1 = 1;     // real_d <- int_d
  real td2 = 2.0;   // real_d <- real_d
  real td_a1[3];
  real td_a2[3] = td_a1;  // real_arr_d <- real_arr_d
  td1 = 2;

  // local variables
  {
    real ltd1 = 1;      // real_d <- int_d 
    real ltd2 = 2.0;    // real_d <- real_d
      real ltd3[4];
      real ltd4[4] = ltd3;
      ltd1 = 5;
  }
}
transformed parameters {
  // block variables
  real tp1 = 1;       // real_p <- int_d 
  real tp2 = 2.0;     // real_p <- real_d
  real tp3 = tp2;      // real_p <- real_p
  real tp4[5];
  real tp5[5] = tp4;
  tp3 = tp1 + tp2;

  // local variables
  {
    real lp1 = 1;      // real_p <- int_d 
    real lp2 = 2.0;    // real_p <- real_d
    real lp3 = tp2;      // real_p <- real_p
      real lp4[6];
      real lp5[6] = lp4;
      lp3 = lp1 + lp2;
  }
}
model {
  // local variables
  real lm1 = 1;      // real_p <- int_d 
  real lm2 = 2.0;    // real_p <- real_d
  real lm3 = tp2;      // real_p <- real_p
  lm3 = lm1 + lm2;
}
generated quantities {
  real gq1 = 1;     // real_d <- int_d
  real gq2 = 2.0;   // real_d <- real_d
  real gq3[7];
  real gq4[7] = gq3;
  gq2 = 5.0;
  {
    real lgq1 = 1;     // real_d <- int_d
    real lgq2 = 2.0;   // real_d <- real_d
    real lgq3;
    real lgq4[8];
    real lgq5[8] = lgq4;
    lgq3 = lgq1 + lgq2;
  }
}
