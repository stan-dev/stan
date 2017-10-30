functions {
  vector target(vector y, vector theta, real[] x_r, int[] x_i) {
    vector[2] deltas;
    deltas[1] = y[1] - theta[1] - x_r[1];
    return deltas;
  }
}

transformed data {
  vector[1] td_y;
  {
    vector[1] td_y_guess = [1]';
    vector[1] td_theta = [1]';
    real td_x_r[0] = {1.0};
    int td_x_i[0];
    td_y = algebra_solver(target, td_y_guess, td_theta, td_x_r, td_x_i);
  }
}
generated quantities {
  vector[1] gq_y;
  {
    vector[1] gq_y_guess = [1]';
    vector[1] gq_theta = [1]';
    real gq_x_r[0] = {1.0};
    int gq_x_i[0];
    gq_y = algebra_solver(target, gq_y_guess, gq_theta, gq_x_r, gq_x_i);
  }
}
