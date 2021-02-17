functions {
  array[] int f1_arr_int() {
    return {-1, -2};
  }
  array[] int f2_arr_int(int x) {
    return {-x};
  }
  array[] real f1_arr_real() {
    return {-1.0, -2.0};
  }
  array[] real f2_arr_real(real y) {
    return {-y};
  }
}
data {
  int d_i1;
  int d_i2;
  int d_i3;
  real d_r1;
  real d_r2;
  real d_r3;
}
transformed data {
  array[3] int td_arr_int_d1_1 = {1, 2, 3};
  array[3] int td_arr_int_d1_2 = {d_i1, 2, 3};
  array[2] int td_arr_int_d1_3 = f1_arr_int();
  array[1] int td_arr_int_d1_4 = f2_arr_int(d_i2);
  array[1, 2] int td_arr_int_d2_1 = {{4, 5}};
  array[1, 2] int td_arr_int_d2_2 = {{4, d_i3}};
  array[2, 3] int td_arr_int_d2_3 = {{1, 2, 3}, {4, 5, 6}};
  array[2, 3] int td_arr_int_d2_4 = {td_arr_int_d1_1, td_arr_int_d1_2};
  array[3] real td_arr_real_d1_1 = {1.1, 2.2, 3.3};
  array[3] real td_arr_real_d1_2 = {d_r1, 2, 3};
  array[2] real td_arr_real_d1_3 = f1_arr_real();
  array[1] real td_arr_real_d1_4 = f2_arr_real(d_r2);
  array[1, 2] real td_arr_real_d2_1 = {{4.4, 5.5}};
  array[1, 2] real td_arr_real_d2_2 = {{4.4, d_r2}};
  print("td_arr_int_d1_1 = ", td_arr_int_d1_1);
  print("td_arr_int_d1_2 = ", td_arr_int_d1_2);
  print("td_arr_int_d1_3 = ", td_arr_int_d1_3);
  print("td_arr_int_d1_4 = ", td_arr_int_d1_4);
  print("td_arr_int_d2_1 = ", td_arr_int_d2_1);
  print("td_arr_int_d2_2 = ", td_arr_int_d2_2);
  print("td_arr_int_d2_3 = ", td_arr_int_d2_3);
  print("td_arr_int_d2_4 = ", td_arr_int_d2_4);
  print("");
  print("td_arr_real_d1_1 = ", td_arr_real_d1_1);
  print("td_arr_real_d1_2 = ", td_arr_real_d1_2);
  print("td_arr_real_d1_3 = ", td_arr_real_d1_3);
  print("td_arr_real_d1_4 = ", td_arr_real_d1_4);
  print("td_arr_real_d2_1 = ", td_arr_real_d2_1);
  print("td_arr_real_d2_2 = ", td_arr_real_d2_2);
  print("");
  {
    array[3] int loc_td_arr_int_d1_1 = {1, 2, 3};
    array[3] int loc_td_arr_int_d1_2 = {d_i1, 2, 3};
    array[2] int loc_td_arr_int_d1_3 = f1_arr_int();
    array[1] int loc_td_arr_int_d1_4 = f2_arr_int(d_i2);
    array[1, 2] int loc_td_arr_int_d2_1 = {{4, 5}};
    array[1, 2] int loc_td_arr_int_d2_2 = {{4, d_i3}};
    array[2, 3] int loc_td_arr_int_d2_3 = {{1, 2, 3}, {4, 5, 6}};
    array[2, 3] int loc_td_arr_int_d2_4 = {loc_td_arr_int_d1_1,
                                           loc_td_arr_int_d1_2};
    array[3] real loc_td_arr_real_d1_1 = {1.1, 2.2, 3.3};
    array[3] real loc_td_arr_real_d1_2 = {d_r1, 2, 3};
    array[2] real loc_td_arr_real_d1_3 = f1_arr_real();
    array[1] real loc_td_arr_real_d1_4 = f2_arr_real(d_r2);
    array[1, 2] real loc_td_arr_real_d2_1 = {{4.4, 5.5}};
    array[1, 2] real loc_td_arr_real_d2_2 = {{4.4, d_r2}};
    print("loc_td_arr_int_d1_1 = ", loc_td_arr_int_d1_1);
    print("loc_td_arr_int_d1_2 = ", loc_td_arr_int_d1_2);
    print("loc_td_arr_int_d1_3 = ", loc_td_arr_int_d1_3);
    print("loc_td_arr_int_d1_4 = ", loc_td_arr_int_d1_4);
    print("loc_td_arr_int_d2_1 = ", loc_td_arr_int_d2_1);
    print("loc_td_arr_int_d2_2 = ", loc_td_arr_int_d2_2);
    print("loc_td_arr_int_d2_3 = ", loc_td_arr_int_d2_3);
    print("loc_td_arr_int_d2_4 = ", loc_td_arr_int_d2_4);
    print("");
    print("loc_td_arr_real_d1_1 = ", loc_td_arr_real_d1_1);
    print("loc_td_arr_real_d1_2 = ", loc_td_arr_real_d1_2);
    print("loc_td_arr_real_d1_3 = ", loc_td_arr_real_d1_3);
    print("loc_td_arr_real_d1_4 = ", loc_td_arr_real_d1_4);
    print("loc_td_arr_real_d2_1 = ", loc_td_arr_real_d2_1);
    print("loc_td_arr_real_d2_2 = ", loc_td_arr_real_d2_2);
    print("");
  }
}
parameters {

}
transformed parameters {
  real tp1 = 0.1;
  real tp2 = 0.2;
  array[3] real tp_arr_real_d1_1 = {1.1, 2.2, 3.3};
  array[3] real tp_arr_real_d1_2 = {d_r1, 2, 3};
  array[2] real tp_arr_real_d1_3 = f1_arr_real();
  array[1] real tp_arr_real_d1_4 = f2_arr_real(d_r2);
  array[2] real tp_arr_real_d1_5 = {tp1, tp2};
  array[1, 2] real tp_arr_real_d2_1 = {{4.4, 5.5}};
  array[1, 2] real tp_arr_real_d2_2 = {{4.4, d_r2}};
  array[1, 2] real tp_arr_real_d2_3 = {{tp1, tp2}};
  print("tp_arr_real_d1_1 = ", tp_arr_real_d1_1);
  print("tp_arr_real_d1_2 = ", tp_arr_real_d1_2);
  print("tp_arr_real_d1_3 = ", tp_arr_real_d1_3);
  print("tp_arr_real_d1_4 = ", tp_arr_real_d1_4);
  print("tp_arr_real_d1_5 = ", tp_arr_real_d1_5);
  print("tp_arr_real_d2_1 = ", tp_arr_real_d2_1);
  print("tp_arr_real_d2_2 = ", tp_arr_real_d2_2);
  print("tp_arr_real_d2_3 = ", tp_arr_real_d2_3);
  print("");
}
model {

}
generated quantities {
  array[3] int gq_arr_int_d1_1 = {1, 2, 3};
  array[3] int gq_arr_int_d1_2 = {d_i1, 2, 3};
  array[2] int gq_arr_int_d1_3 = f1_arr_int();
  array[1] int gq_arr_int_d1_4 = f2_arr_int(d_i2);
  array[1, 2] int gq_arr_int_d2_1 = {{4, 5}};
  array[1, 2] int gq_arr_int_d2_2 = {{4, d_i3}};
  array[2, 3] int gq_arr_int_d2_3 = {{1, 2, 3}, {4, 5, 6}};
  array[2, 3] int gq_arr_int_d2_4 = {gq_arr_int_d1_1, gq_arr_int_d1_2};
  array[3] real gq_arr_real_d1_1 = {1.1, 2.2, 3.3};
  array[3] real gq_arr_real_d1_2 = {d_r1, 2, 3};
  array[2] real gq_arr_real_d1_3 = f1_arr_real();
  array[1] real gq_arr_real_d1_4 = f2_arr_real(d_r2);
  array[1, 2] real gq_arr_real_d2_1 = {{4.4, 5.5}};
  array[1, 2] real gq_arr_real_d2_2 = {{4.4, d_r2}};
  print("gq_arr_int_d1_1 = ", gq_arr_int_d1_1);
  print("gq_arr_int_d1_2 = ", gq_arr_int_d1_2);
  print("gq_arr_int_d1_3 = ", gq_arr_int_d1_3);
  print("gq_arr_int_d1_4 = ", gq_arr_int_d1_4);
  print("gq_arr_int_d2_1 = ", gq_arr_int_d2_1);
  print("gq_arr_int_d2_2 = ", gq_arr_int_d2_2);
  print("gq_arr_int_d2_3 = ", gq_arr_int_d2_3);
  print("gq_arr_int_d2_4 = ", gq_arr_int_d2_4);
  print("");
  print("gq_arr_real_d1_1 = ", gq_arr_real_d1_1);
  print("gq_arr_real_d1_2 = ", gq_arr_real_d1_2);
  print("gq_arr_real_d1_3 = ", gq_arr_real_d1_3);
  print("gq_arr_real_d1_4 = ", gq_arr_real_d1_4);
  print("gq_arr_real_d2_1 = ", gq_arr_real_d2_1);
  print("gq_arr_real_d2_2 = ", gq_arr_real_d2_2);
  print("");
  {
    array[3] int loc_gq_arr_int_d1_1 = {1, 2, 3};
    array[3] int loc_gq_arr_int_d1_2 = {d_i1, 2, 3};
    array[2] int loc_gq_arr_int_d1_3 = f1_arr_int();
    array[1] int loc_gq_arr_int_d1_4 = f2_arr_int(d_i2);
    array[1, 2] int loc_gq_arr_int_d2_1 = {{4, 5}};
    array[1, 2] int loc_gq_arr_int_d2_2 = {{4, d_i3}};
    array[2, 3] int loc_gq_arr_int_d2_3 = {{1, 2, 3}, {4, 5, 6}};
    array[2, 3] int loc_gq_arr_int_d2_4 = {loc_gq_arr_int_d1_1,
                                           loc_gq_arr_int_d1_2};
    array[3] real loc_gq_arr_real_d1_1 = {1.1, 2.2, 3.3};
    array[3] real loc_gq_arr_real_d1_2 = {d_r1, 2, 3};
    array[2] real loc_gq_arr_real_d1_3 = f1_arr_real();
    array[1] real loc_gq_arr_real_d1_4 = f2_arr_real(d_r2);
    array[1, 2] real loc_gq_arr_real_d2_1 = {{4.4, 5.5}};
    array[1, 2] real loc_gq_arr_real_d2_2 = {{4.4, d_r2}};
    print("loc_gq_arr_int_d1_1 = ", loc_gq_arr_int_d1_1);
    print("loc_gq_arr_int_d1_2 = ", loc_gq_arr_int_d1_2);
    print("loc_gq_arr_int_d1_3 = ", loc_gq_arr_int_d1_3);
    print("loc_gq_arr_int_d1_4 = ", loc_gq_arr_int_d1_4);
    print("loc_gq_arr_int_d2_1 = ", loc_gq_arr_int_d2_1);
    print("loc_gq_arr_int_d2_2 = ", loc_gq_arr_int_d2_2);
    print("");
    print("loc_gq_arr_real_d1_1 = ", loc_gq_arr_real_d1_1);
    print("loc_gq_arr_real_d1_2 = ", loc_gq_arr_real_d1_2);
    print("loc_gq_arr_real_d1_3 = ", loc_gq_arr_real_d1_3);
    print("loc_gq_arr_real_d1_4 = ", loc_gq_arr_real_d1_4);
    print("loc_gq_arr_real_d2_1 = ", loc_gq_arr_real_d2_1);
    print("loc_gq_arr_real_d2_2 = ", loc_gq_arr_real_d2_2);
    print("");
  }
}

