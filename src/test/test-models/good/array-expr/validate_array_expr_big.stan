transformed data {
  int i = 5;
  int td_ar_int_dim1[100]
      = { 1, 2, 3, 5, i, i, 7, 8, i, 10,
          1, 2, 3, 5, i, i, 7, 8, i, 10,
          1, 2, 3, 5, i, i, 7, 8, i, 10,
                             1, 2, 3, 5, i, i, 7, 8, i, 10,
          1, 2, 3, 5, i, i, 7, 8, i, 10,
          1, 2, 3, 5, i, i, 7, 8, i, 10,
          1, 2, 3, 5, i, i, 7, 8, i, 10,
          1, 2, 3, 5, i, i, 7, 8, i, 10,
          1, 2, 3, 5, i, i, 7, 8, i, 10 };
}
parameters {
  real<lower=0,upper=1> theta;
}
