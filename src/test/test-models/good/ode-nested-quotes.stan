functions {
  array[] real foo(real t, array[] real y, array[] real theta,
                   array[] real x_r, array[] int x_i) {
    return rep_array(1.0, 1);
  }
}
transformed data {
  real y;
  array[2, 2] real t;
  y = integrate_ode(foo, rep_array(1.0, 1), 1.0, t[1], rep_array(1.0, 1),
                    rep_array(1.0, 1), rep_array(1, 1))[1, 1];
}
model {

}

