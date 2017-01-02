functions {
  void foo() {
    vector[-1] lf1;
    vector[1] lf2[-1];
  }
}
data {
  int veclen;
  int arrlen;
  int nrows;
  int ncols;
  int neg_d;
  real d_real_ar[arrlen];
  vector[veclen] d_v;
  row_vector[veclen] d_rv;
  vector[veclen] d_v_ar[arrlen];
  row_vector[veclen] d_rv_ar[arrlen];
  matrix[nrows, ncols] d_m;
}
transformed data {
  real td_real_ar[arrlen] = d_real_ar;
  vector[veclen] td_v1 = d_v;
  vector[veclen] td_v_ar[arrlen] = d_v_ar;

  {
    vector[veclen] local_v1 = d_v;
    vector[veclen] local_v_ar[arrlen] = d_v_ar;

    row_vector[veclen] local_rv1 = d_rv;
    row_vector[veclen] local_rv_ar[arrlen] = d_rv_ar;
  }
}
parameters {
  matrix<lower = 0, upper = 1>[nrows, ncols] p_m;
  real<lower=-10, upper=10> y[2];
  vector[-1] theta;
}
transformed parameters {
  real tp_real_ar[arrlen] = td_real_ar;
  vector[veclen] tp_v1 = d_v;
  vector[veclen] tp_v_ar4[arrlen] = d_v_ar;

  row_vector[veclen] tp_rv1 = d_rv;
  row_vector[veclen] tp_rv_ar8[arrlen];

  {
    real local_real_ar[arrlen] = d_real_ar;
    vector[veclen] local_v1 = d_v;
    vector[veclen] local_v_ar4[arrlen] = d_v_ar;

    row_vector[veclen] local_rv1 = d_rv;
    row_vector[veclen] local_rv_ar8[arrlen];
  }
}
model {
  y ~ normal(0,1);
}
generated quantities {
}
