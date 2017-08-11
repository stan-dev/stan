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
  vector[2] y;
}

transformed data {
  real x_r[0];
  int x_i[0];
}

parameters {
  vector[2] theta;
  real dummy_parameter;
}

transformed parameters {
}

model {
  dummy_parameter ~ normal(0, 1);
}

generated quantities {
  vector[2] y_s;
  y_s = algebra_solver(algebra_system, y, theta, x_r, x_i);
}
