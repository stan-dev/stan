transformed data {
  int td_ar_int_dim1[3];
  real td_ar_real_dim1[3];
  real td_ar_real_dim2[2,3];
  real td_x;
  real td_y;
  real td_z;
  td_ar_int_dim1 = {1, 2, 3};
  td_ar_real_dim1 = {1.0, 2*2, 3^2};
  td_ar_real_dim1 = {td_x,td_y,td_z};
  td_ar_real_dim2 = { {1.0, 2*2, 3^2}, {4.0, 5.0, 6.0} };
}
parameters {
  real a;
}
transformed parameters {
  real tp_ar_real_dim1[3];
  real tp_x;
  real tp_y;
  real tp_z;
  tp_ar_real_dim1 = {1.0, 2*2, 3^2};
  tp_ar_real_dim1 = {tp_x,tp_y,tp_z};
}
model {
  a ~ normal(0,1);
  print({1.0, 2.0, 3.0});
}
