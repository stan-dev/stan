functions {
  real algebra_solverfake(real x) {
    return 2 * x;
  }
  vector algebra_system(vector x,
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

}

transformed data {
  vector[2] x;
  vector[2] y;
  real dat[0];
  int dat_int[0];
  vector[2] theta;

  theta = algebra_solver(algebra_system, x, y, dat, dat_int);
  theta = algebra_solver(algebra_system, x, y, dat, dat_int, 0.01, 0.01, 10);
}

parameters {
  vector[2] x_p;
  vector[2] y_p;
  real dummy_parameter;
}

transformed parameters {
  real abc_tp = algebra_solverfake(2.9);
  vector[2] theta_p;

  theta_p = algebra_solver(algebra_system, x, y, dat, dat_int);
  theta_p = algebra_solver(algebra_system, x, y, dat, dat_int, 0.01, 0.01, 10);

  theta_p = algebra_solver(algebra_system, x, y_p, dat, dat_int);
  theta_p = algebra_solver(algebra_system, x, y_p, dat, dat_int, 0.01, 0.01, 10);

  theta_p = algebra_solver(algebra_system, x_p, y, dat, dat_int);
  theta_p = algebra_solver(algebra_system, x_p, y, dat, dat_int, 0.01, 0.01, 10);

  theta_p = algebra_solver(algebra_system, x_p, y_p, dat, dat_int);
  theta_p = algebra_solver(algebra_system, x_p, y_p, dat, dat_int, 0.01, 0.01, 10);


}

model {
  dummy_parameter ~ normal(0, 1);
}
