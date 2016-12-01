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
} 
transformed data {
  int td0 = 3;
  real td1 = 1;     // real_d <- int_d
  real td2 = 2.0;   // real_d <- real_d
  real td3 = td0;
  real td_a1[3];
  real td_a2[3] = td_a1;  // real_arr_d <- real_arr_d
  print("td1: ",td1);
  print("td2: ",td2);
  print("td3: ",td3);
  print("td_a2: ",td_a2);
  td1 = foo(td3);
  print("td1: ",td1);
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
  real tp1 = 1;       // real_p <- int_d 
  real tp2 = 2.0;     // real_p <- real_d
  real tp3 = tp2;      // real_p <- real_p
  real tp4[td0];
  real tp5[td0] = tp4;
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
  real gq1 = 1;     // real_d <- int_d
  real gq2 = 2.0;   // real_d <- real_d
  real gq3[td0];
  real gq4[td0] = gq3;
  print("gq1: ",gq1);
  print("gq2: ",gq2);
  print("gq4: ",gq4);
  {
    real lgq1 = 1;     // real_d <- int_d
    real lgq2 = 2.0;   // real_d <- real_d
    real lgq3[td0];
    real lgq4[td0] = lgq3;
    print("lgq1: ",lgq1);
    print("lgq2: ",lgq2);
    print("lgq4: ",lgq4);
  }
}
