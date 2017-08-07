functions {
  real my_fun(real x);
  real my_fun2(real x, real y);
  real my_fun3(data real x);
  real my_fun(real x) {
    return 2 * x;
  }
  real my_fun2(real x, real y) {
    return 2 * x;
  }
  real my_fun3(data real x) {
    return 2 * x;
  }
}
transformed data {
  real td_d1 = 1;
  real td_d2 = my_fun(td_d1);
  real td_d3 = my_fun2(td_d1, td_d2);
  td_d3 = my_fun3(my_fun2(td_d1, td_d2));
}
parameters {
  real p_d1;
}
transformed parameters {
  real tp_d1 = my_fun(p_d1);
  real tp_d2 = my_fun2(p_d1, tp_d1);
}
generated quantities {
  real gq_d1 = my_fun(p_d1);
  real gq_d2 = my_fun(gq_d1);
  gq_d2 = my_fun3(gq_d1);
}
