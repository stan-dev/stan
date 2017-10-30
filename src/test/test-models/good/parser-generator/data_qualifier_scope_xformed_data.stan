functions {
  vector target(vector y, vector theta, real[] x_r, int[] x_i) {
    vector[2] deltas;
    deltas[1] = y[1] - theta[1] - x_r[1];
    return deltas;
  }
}

transformed data {
  vector[1] y;
  {
    vector[1] y_guess = [1]';
    vector[1] theta = [1]';
    real x_r[0] = {1.0};
    int x_i[0];

    y = algebra_solver(tail_delta, y_guess, theta, x_r, x_i);
  }
}
