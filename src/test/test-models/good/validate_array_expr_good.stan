functions {
  real[] f1(real y) {
    real x = 1 ? y : 2;
    return  {1.0, 2.0, y};
  }
  int[] f2(int y) {
    return  {1, 2, 3};
  }
  real[] f3(real y) {
    real x = 1;
    return {x, y};
  }
}
data { 
  int<lower=0> N; 
  int<lower=0,upper=1> y[N];
} 
transformed data {
  int td_ar_int_dim1[3] = {1, 2, 3};
  real td_ar_real_dim1_1[1] = { 1.0 };
  real td_ar_real_dim1_2[2] = { 1.1, 2.1 };
  real td_ar_real_dim2_23[2,3] = {{1.2,2.2,2.3},{4.2,5.2,6.2}};
  real td_ar_real_dim3_123[1,2,3] = {{{1.2,2.2,2.3},{4.2,5.2,6.2}}};
  real a = 1.0;
  td_ar_real_dim1_2 = f3(a);
  {
    real loc_td_ar_real_dim1_2[2] = { 1.1, 2.1 };
    real loc_td_ar_real_dim2_23[2,3] = {{1.2,2.2,2.3},{4.2,5.2,6.2}};
    loc_td_ar_real_dim1_2 = f3(a);
  }
}
parameters {
  real<lower=0,upper=1> theta;
} 
transformed parameters {
  real x = 1.1;
  real tp_ar_real_dim1_1[1] = {x};
  real tp_ar_real_dim1_2[2] = {theta, x};
  real tp_ar_real_dim1_3[3] = { 1.0*4.5, x*4.5, 2.2^4.5};
  real tp_ar_real_dim1_4[4] = { 1.0*4.5, a*4.5, 2.2^4.5, a};
  tp_ar_real_dim1_2 = f3(x);
  {
    real loc_tp_ar_real_dim1_2[2] = {theta, x};
  }
}
model {
  theta ~ beta(1,1);
  for (n in 1:N) 
    y[n] ~ bernoulli(theta);
}
generated quantities {
  real gq_ar_real_dim1_1[1] = {x};
  real gq_ar_real_dim1_2[2] = {theta, x};
  real gq_ar_real_dim1_3[3] = { 1.0*4.5, x*4.5, 2.2^4.5};
  gq_ar_real_dim1_2 = f3(x);
  {
    real loc_gq_ar_real_dim1_2[2] = {theta, x};
    loc_gq_ar_real_dim1_2 = f3(x);
  }
}
