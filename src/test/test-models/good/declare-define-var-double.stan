functions {
  real foo(real a1) {
    int lf0 = 2;
    real lf1 = a1;
    real lf2 = lf1;
    real lf3[lf0];
    real lf4[lf0] = lf3;
    print("foo, lf1: ", lf1);
    print("foo, lf2: ", lf1);
    print("foo, lf4: ", lf4);
    lf1 = lf3[1] + lf4[1];
    return lf1;
  }
}
data { 
  int<lower=0> N; 
  int<lower=0,upper=1> y[N];
  real d[3,3];
} 
transformed data {
  int td0 = 3;
  real td1 = 123;     // real_d <- int_d
  real td2 = 2.0;   // real_d <- real_d
  real td3 = td0;
  real td4 = td3;
  real td_a1[3];
  real td_a2[3] = td_a1;  // real_arr_d <- real_arr_d
  real td5 = td_a2[1];
  real td_a3[3, 3] = d;
  real td_a4[3] = td_a3[2];
  print("td1: ",td1);
  print("td2: ",td2);
  print("td3: ",td3);
  print("td4: ",td4);
  print("td5: ",td5);
  print("td_a2: ",td_a2);
  print("td_a4: ",td_a4);
  {
    real ltd1 = 1;      // real_d <- int_d 
    real ltd2 = 2.0;    // real_d <- real_d
    real ltd3 = td1;
    real ltd4[td0];
    real ltd5[td0] = ltd4;
    print("ltd1: ",ltd1);
    print("ltd2: ",ltd2);
    print("ltd3: ",ltd3);
    print("ltd5: ",ltd5);
  }
}
parameters {
  real<lower=0,upper=1> theta;
} 
transformed parameters {
  real d_tp1 = 1.0;
  real d_tp2 = td1;
  real d_tp3 = td0;
  real d_tp4 = d[1,2];
  real d_tp_a1[3] = td_a4;
  real d_tp_a2[3] = td_a3[1];
  real d_tp_a3[3, 3] = d;

  real p_tp2 = d_tp1;
  real p_tp4 = d_tp_a1[1];
  real p_tp_a1[3] = d_tp_a1;
  real p_tp_a2[3] = d_tp_a3[3];


  real tp1 = 1;       // real_p <- int_d 
  real tp2 = 2.0;     // real_p <- real_d
  real tp3 = tp2;      // real_p <- real_p
  real tp4[td0];
  real tp5[td0] = tp4;

  print("d_tp1 = ", d_tp1);
  print("d_tp2 = ", d_tp2, " should be td1 = ", td1, " which should be 123");
  print("d_tp3 = ", d_tp3);
  print("d_tp4 = ", d_tp4);
  print("d_tp_a1 = ", d_tp_a1);
  print("d_tp_a2 = ", d_tp_a2);
  print("d_tp_a3 = ", d_tp_a3);

  print("p_tp2 = ", p_tp2);
  print("p_tp4 = ", p_tp4);
  print("p_tp_a1 = ", p_tp_a1);
  print("p_tp_a2 = ", p_tp_a2);
  
  print("tp1: ",tp1);
  print("tp2: ",tp2);
  print("tp3: ",tp3);
  print("tp5: ",tp5);
  tp1 = foo(tp3);
  print("tp1: ",tp1);
  {
    real lp1 = td0;      // real_p <- int_d 
    real lp2 = 9.0;    // real_p <- real_d
    real lp3 = tp2;      // real_p <- real_p
    real lp4[td0];
    real lp5[td0] = lp4;
    print("lp1: ",lp1);
    print("lp2: ",lp2);
    print("lp3: ",lp3);
    print("lp5: ",lp5);
  }
}
model {
  // local variables
  real lm1 = 1;      // real_p <- int_d 
  real lm2 = 2.0;    // real_p <- real_d
  real lm3 = tp2;      // real_p <- real_p
  theta ~ beta(1,1);
  for (n in 1:N) 
    y[n] ~ bernoulli(theta);
}
generated quantities {
  real gq_d_tp1 = 1.0;
  real gq_d_tp2 = td1;
  real gq_d_tp3 = td0;
  real gq_d_tp4 = d[1,2];
  real gq_d_tp_a1[3] = td_a4;
  real gq_d_tp_a2[3] = td_a3[1];
  real gq_d_tp_a3[3, 3] = d;

  real gq_p_tp2 = d_tp1;
  real gq_p_tp4 = d_tp_a1[1];
  real gq_p_tp_a1[3] = d_tp_a1;
  real gq_p_tp_a2[3] = d_tp_a3[3];
  print("gq_d_tp1 = ", gq_d_tp1);
  print("gq_d_tp2 = ", gq_d_tp2, " should be td1 = ", td1, " which should be 123");
  print("gq_d_tp3 = ", gq_d_tp3);
  print("gq_d_tp4 = ", gq_d_tp4);
  print("gq_d_tp_a1 = ", gq_d_tp_a1);
  print("gq_d_tp_a2 = ", gq_d_tp_a2);
  print("gq_d_tp_a3 = ", gq_d_tp_a3);

  print("gq_p_tp2 = ", gq_p_tp2);
  print("gq_p_tp4 = ", gq_p_tp4);
  print("gq_p_tp_a1 = ", gq_p_tp_a1);
  print("gq_p_tp_a2 = ", gq_p_tp_a2);
  {
    real lgq1 = 1;     // real_d <- int_d
    real lgq2 = 2.0;   // real_d <- real_d
    real lqd2a = lgq2;
    real lgq3[td0];
    real lgq3a = lgq3[1];
    real lgq4[td0] = lgq3;
    real lgq5[3] = d[1];
    print("lgq1: ",lgq1);
    print("lgq2: ",lgq2);
    print("lgq2a: ",lqd2a);
    print("lgq3: ",lgq3);
    print("lgq3a: ",lgq3a);
    print("lgq4: ",lgq4);
    print("lgq5: ",lgq5);
  }
}
