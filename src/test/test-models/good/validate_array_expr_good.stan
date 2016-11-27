functions {
  real[] f1(real y) {
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
  vector[7] b0;
  row_vector[7] c0;
  real td_x;
  real td_y;
  real td_z;
  int td_ar_int_dim1[3] = {1, 2, 3};
  //  real td_ar_real_dim2[2,3];
  //  td_ar_real_dim2 = { {1.0, 2*2, 3^2}, {4.0, 5.0, 6.0} };
  //  real td_ar_real_dim1[3] = {1.0, 2*2, 3^2};
  //  real td_ar_real_dim1_s1[1] = {td_z};
  //  real td_ar_real_dim1_s4[4] = {1.1, 2.0, 3.0}; // cannot check length
  //  real td_ar_real_dim1_s3[3] = {1.1, 2.0, 3.0, 4.0, 5.0}; // cannot check length

}
parameters {
  real<lower=0,upper=1> theta;
} 
transformed parameters {
  real tp_x;
  real tp_y;
  real tp_z;
  real tp_ar_real_dim1[3];
  //  real tp_ar_real_dim2[2,3];
  tp_ar_real_dim1 = {1.0, 2*2, 3^2};
  tp_ar_real_dim1 = {tp_x,tp_y,tp_z};
  tp_ar_real_dim1 = f1(tp_z);
}
model {
  theta ~ beta(1,1);
  for (n in 1:N) 
    y[n] ~ bernoulli(theta);
  print({1.0, 2.0, 3.0});
}
