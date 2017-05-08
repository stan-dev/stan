functions {
  vector algebra_system (vector x,
                         vector y,
                         real[] dat,
                         int[] dat_int) {
    vector[2] f_x;
    f_x[1] = x[1] - y[1];
    f_x[2] = x[2] - y[2];
    return f_x;
  }
}


data {
  // vector[2] x;
  // vector[2] y;
  // real dat[0];
  // int dat_int[0];
  // vector[2] theta;
}

transformed data {
  vector[2] x;
  vector[2] y;
  real dat[0];
  int dat_int[0];
  vector[2] theta;

  theta = algebra_solver(algebra_system, x, y, dat, dat_int);
}

parameters {
  vector[2] y_p;
  real dummy_parameter;
}

transformed parameters {
  vector[2] theta_p;
  theta_p = algebra_solver(algebra_system, x, y, dat, dat_int);
  theta_p = algebra_solver(algebra_system, x, y_p, dat, dat_int);
}

model {
  dummy_parameter ~ normal(0, 1);
}

