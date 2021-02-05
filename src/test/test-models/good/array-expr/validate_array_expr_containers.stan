functions {

}
data {
  int d;
  vector[d] d_v1;
  vector[d] d_v2;
  row_vector[d] d_rv1;
  row_vector[d] d_rv2;
  array[d] vector[d] d_v_ar;
  array[d] row_vector[d] d_rv_ar;
  matrix[d, d] d_m;
}
transformed data {
  array[1] vector[d] td_v_ar_dim1_1 = {d_v1};
  array[2] vector[d] td_v_ar_dim1_2 = {d_v1, d_v2};
  array[1] vector[d] td_v_ar_dim1_3 = {d_v_ar[3]};
  array[1] vector[d] td_v_ar_dim1_4 = {d_m[ : , 3]};
  array[1, 1] vector[d] td_v_ar_dim2_1 = {{d_v1}};
  array[2, 3] vector[d] td_v_ar_dim2_2 = {{d_v1, d_v1, d_v1},
                                          {d_v2, d_v2, d_v2}};
  array[1] row_vector[d] td_rv_ar_dim1_1 = {d_rv1};
  array[2] row_vector[d] td_rv_ar_dim1_2 = {d_rv1, d_rv2};
  array[1] row_vector[d] td_rv_ar_dim1_3 = {d_rv_ar[3]};
  array[1, 1] row_vector[d] td_rv_ar_dim2_1 = {{d_rv1}};
  array[2, 3] row_vector[d] td_rv_ar_dim2_2 = {{d_rv1, d_rv1, d_rv1},
                                               {d_rv2, d_rv2, d_rv2}};
  array[1] matrix[d, d] td_m_ar_dim1_1 = {d_m};
  array[2] matrix[d, d] td_m_ar_dim1_2 = {d_m, d_m};
  array[1, 1] matrix[d, d] td_m_ar_dim2_1 = {{d_m}};
  array[2, 3] matrix[d, d] td_m_ar_dim2_2 = {{d_m, d_m, d_m},
                                             {d_m, d_m, d_m}};
  print("td_v_ar_dim1_1 = ", td_v_ar_dim1_1);
  print("td_v_ar_dim1_2 = ", td_v_ar_dim1_2);
  print("td_v_ar_dim1_3 = ", td_v_ar_dim1_3);
  print("td_v_ar_dim1_4 = ", td_v_ar_dim1_4);
  print("td_v_ar_dim2_1 = ", td_v_ar_dim2_1);
  print("td_v_ar_dim2_2 = ", td_v_ar_dim2_2);
  print("td_rv_ar_dim1_1 = ", td_rv_ar_dim1_1);
  print("td_rv_ar_dim1_2 = ", td_rv_ar_dim1_2);
  print("td_rv_ar_dim1_3 = ", td_rv_ar_dim1_3);
  print("td_rv_ar_dim2_1 = ", td_rv_ar_dim2_1);
  print("td_rv_ar_dim2_2 = ", td_rv_ar_dim2_2);
  print("td_m_ar_dim1_1 = ", td_m_ar_dim1_1);
  print("td_m_ar_dim1_2 = ", td_m_ar_dim1_2);
  print("td_m_ar_dim2_1 = ", td_m_ar_dim2_1);
  print("td_m_ar_dim2_2 = ", td_m_ar_dim2_2);
}
parameters {

}
transformed parameters {
  array[1] vector[d] tp_v_ar_dim1_1 = {d_v1};
  array[2] vector[d] tp_v_ar_dim1_2 = {d_v1, d_v2};
  array[1] vector[d] tp_v_ar_dim1_3 = {d_v_ar[3]};
  vector[d] tp_v1 = d_m[ : , 4];
  array[1] vector[d] tp_v_ar_dim1_4 = {tp_v1};
  array[2] vector[d] tp_v_ar_dim1_5 = {d_v1, tp_v1};
  array[1, 1] vector[d] tp_v_ar_dim2_1 = {{d_v1}};
  array[2, 3] vector[d] tp_v_ar_dim2_2 = {{d_v1, d_v1, d_v1},
                                          {d_v2, d_v2, d_v2}};
  array[1] row_vector[d] tp_rv_ar_dim1_1 = {d_rv1};
  array[2] row_vector[d] tp_rv_ar_dim1_2 = {d_rv1, d_rv2};
  array[1] row_vector[d] tp_rv_ar_dim1_3 = {d_rv_ar[3]};
  array[1, 1] row_vector[d] tp_rv_ar_dim2_1 = {{d_rv1}};
  array[2, 3] row_vector[d] tp_rv_ar_dim2_2 = {{d_rv1, d_rv1, d_rv1},
                                               {d_rv2, d_rv2, d_rv2}};
  array[1] matrix[d, d] tp_m_ar_dim1_1 = {d_m};
  array[2] matrix[d, d] tp_m_ar_dim1_2 = {d_m, d_m};
  array[1, 1] matrix[d, d] tp_m_ar_dim2_1 = {{d_m}};
  array[2, 3] matrix[d, d] tp_m_ar_dim2_2 = {{d_m, d_m, d_m},
                                             {d_m, d_m, d_m}};
  print("tp_v_ar_dim1_1 = ", tp_v_ar_dim1_1);
  print("tp_v_ar_dim1_2 = ", tp_v_ar_dim1_2);
  print("tp_v_ar_dim1_3 = ", tp_v_ar_dim1_3);
  print("tp_v_ar_dim1_4 = ", tp_v_ar_dim1_4);
  print("tp_v_ar_dim1_5 = ", tp_v_ar_dim1_5);
  print("tp_v_ar_dim2_1 = ", tp_v_ar_dim2_1);
  print("tp_v_ar_dim2_2 = ", tp_v_ar_dim2_2);
  print("tp_rv_ar_dim1_1 = ", tp_rv_ar_dim1_1);
  print("tp_rv_ar_dim1_2 = ", tp_rv_ar_dim1_2);
  print("tp_rv_ar_dim1_3 = ", tp_rv_ar_dim1_3);
  print("tp_rv_ar_dim2_1 = ", tp_rv_ar_dim2_1);
  print("tp_rv_ar_dim2_2 = ", tp_rv_ar_dim2_2);
  print("tp_m_ar_dim1_1 = ", tp_m_ar_dim1_1);
  print("tp_m_ar_dim1_2 = ", tp_m_ar_dim1_2);
  print("tp_m_ar_dim2_1 = ", tp_m_ar_dim2_1);
  print("tp_m_ar_dim2_2 = ", tp_m_ar_dim2_2);
}
model {

}
generated quantities {
  array[1] vector[d] gq_v_ar_dim1_1 = {d_v1};
  array[2] vector[d] gq_v_ar_dim1_2 = {d_v1, d_v2};
  array[1] vector[d] gq_v_ar_dim1_3 = {d_v_ar[3]};
  array[1] vector[d] gq_v_ar_dim1_4 = {d_m[ : , 3]};
  array[1, 1] vector[d] gq_v_ar_dim2_1 = {{d_v1}};
  array[2, 3] vector[d] gq_v_ar_dim2_2 = {{d_v1, d_v1, d_v1},
                                          {d_v2, d_v2, d_v2}};
  array[1] row_vector[d] gq_rv_ar_dim1_1 = {d_rv1};
  array[2] row_vector[d] gq_rv_ar_dim1_2 = {d_rv1, d_rv2};
  array[1] row_vector[d] gq_rv_ar_dim1_3 = {d_rv_ar[3]};
  array[1, 1] row_vector[d] gq_rv_ar_dim2_1 = {{d_rv1}};
  array[2, 3] row_vector[d] gq_rv_ar_dim2_2 = {{d_rv1, d_rv1, d_rv1},
                                               {d_rv2, d_rv2, d_rv2}};
  array[1] matrix[d, d] gq_m_ar_dim1_1 = {d_m};
  array[2] matrix[d, d] gq_m_ar_dim1_2 = {d_m, d_m};
  array[1, 1] matrix[d, d] gq_m_ar_dim2_1 = {{d_m}};
  array[2, 3] matrix[d, d] gq_m_ar_dim2_2 = {{d_m, d_m, d_m},
                                             {d_m, d_m, d_m}};
  print("gq_v_ar_dim1_1 = ", gq_v_ar_dim1_1);
  print("gq_v_ar_dim1_2 = ", gq_v_ar_dim1_2);
  print("gq_v_ar_dim1_3 = ", gq_v_ar_dim1_3);
  print("gq_v_ar_dim1_4 = ", gq_v_ar_dim1_4);
  print("gq_v_ar_dim2_1 = ", gq_v_ar_dim2_1);
  print("gq_v_ar_dim2_2 = ", gq_v_ar_dim2_2);
  print("gq_rv_ar_dim1_1 = ", gq_rv_ar_dim1_1);
  print("gq_rv_ar_dim1_2 = ", gq_rv_ar_dim1_2);
  print("gq_rv_ar_dim1_3 = ", gq_rv_ar_dim1_3);
  print("gq_rv_ar_dim2_1 = ", gq_rv_ar_dim2_1);
  print("gq_rv_ar_dim2_2 = ", gq_rv_ar_dim2_2);
  print("gq_m_ar_dim1_1 = ", gq_m_ar_dim1_1);
  print("gq_m_ar_dim1_2 = ", gq_m_ar_dim1_2);
  print("gq_m_ar_dim2_1 = ", gq_m_ar_dim2_1);
  print("gq_m_ar_dim2_2 = ", gq_m_ar_dim2_2);
}

