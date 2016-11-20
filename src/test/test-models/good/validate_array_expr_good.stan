transformed data {
  int a_ar_int_dim1[3];
  real a_ar_real_dim1[3];
  real x;
  real y;
  real z;
  a_ar_int_dim1 = {1, 2, 3};
  a_ar_real_dim1 = {1.0, 2*2, 3^2};
  a_ar_real_dim1 = {x,y,z};
}
parameters {
  real a;
}
model {
  a ~ normal(0,1);
  print({1.0, 2.0, 3.0});
}
