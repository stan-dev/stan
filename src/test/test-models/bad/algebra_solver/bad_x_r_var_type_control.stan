functions {
  vector algebra_system (vector y,
                         vector theta,
                         real[] x_r,
                         int[] x_i) {
    vector[2] f_y;
    f_y[1] = y[1] - theta[1];
    f_y[2] = y[2] - theta[2];
    return f_y;
  }
}

data {

}

transformed data {
  vector[2] y;
  // real x_r[0];
  int x_i[0];
}

parameters {
  real x_r_p[0];
  vector[2] theta_p;
  real dummy_parameter;
}

transformed parameters {
  vector[2] y_s_p;
  y_s_p = algebra_solver(algebra_system, y, theta_p, x_r_p, x_i, 0.01, 0.01, 10);
}

model {
  dummy_parameter ~ normal(0, 1);
}
