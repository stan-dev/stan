functions {
  vector algebra_system(vector y, vector theta, array[] real x_r,
                        array[] int x_i) {
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
  array[0] real x_r;
  array[0] int x_i;
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

