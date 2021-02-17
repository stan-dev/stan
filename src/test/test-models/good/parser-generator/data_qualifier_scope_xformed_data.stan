functions {
  vector target_v(vector y, vector theta, array[] real x_r, array[] int x_i) {
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
    array[0] real td_x_r = {1.0};
    array[0] int td_x_i;
    td_y = algebra_solver(target_v, td_y_guess, td_theta, td_x_r, td_x_i);
  }
}
generated quantities {
  vector[1] gq_y;
  {
    vector[1] gq_y_guess = [1]';
    vector[1] gq_theta = [1]';
    array[0] real gq_x_r = {1.0};
    array[0] int gq_x_i;
    gq_y = algebra_solver(target_v, gq_y_guess, gq_theta, gq_x_r, gq_x_i);
  }
}

