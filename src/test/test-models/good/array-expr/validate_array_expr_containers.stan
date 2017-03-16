functions {
}
data { 
  int d;
  vector[d] d_v1;
  vector[d] d_v2;
  row_vector[d] d_rv1;
  row_vector[d] d_rv2;
  vector[d] d_v_ar[d];
  row_vector[d] d_rv_ar[d];
  matrix[d, d] d_m;
} 
transformed data {
  vector[d] td_v_ar_dim1_1[1] = { d_v1 };
  vector[d] td_v_ar_dim1_2[2] = { d_v1, d_v2 };
  vector[d] td_v_ar_dim1_3[1] = { d_v_ar[3] };
  vector[d] td_v_ar_dim1_4[1] = { d_m[,3] };
  vector[d] td_v_ar_dim2_1[1,1] = { { d_v1 } };
  vector[d] td_v_ar_dim2_2[2,3] = { { d_v1, d_v1, d_v1 }, { d_v2, d_v2, d_v2 } };

  row_vector[d] td_rv_ar_dim1_1[1] = { d_rv1 };
  row_vector[d] td_rv_ar_dim1_2[2] = { d_rv1, d_rv2 };
  row_vector[d] td_rv_ar_dim1_3[1] = { d_rv_ar[3] };
  row_vector[d] td_rv_ar_dim2_1[1,1] = { { d_rv1 } };
  row_vector[d] td_rv_ar_dim2_2[2,3] = { { d_rv1, d_rv1, d_rv1 }, { d_rv2, d_rv2, d_rv2 } };

  matrix[d,d] td_m_ar_dim1_1[1] = { d_m };
  matrix[d,d] td_m_ar_dim1_2[2] = { d_m, d_m };
  matrix[d,d] td_m_ar_dim2_1[1,1] = { { d_m } };
  matrix[d,d] td_m_ar_dim2_2[2,3] = { { d_m, d_m, d_m }, { d_m, d_m, d_m } };

  print("td_v_ar_dim1_1 = ",td_v_ar_dim1_1);
  print("td_v_ar_dim1_2 = ",td_v_ar_dim1_2);
  print("td_v_ar_dim1_3 = ",td_v_ar_dim1_3);
  print("td_v_ar_dim1_4 = ",td_v_ar_dim1_4);
  print("td_v_ar_dim2_1 = ",td_v_ar_dim2_1);
  print("td_v_ar_dim2_2 = ",td_v_ar_dim2_2);

  print("td_rv_ar_dim1_1 = ",td_rv_ar_dim1_1);
  print("td_rv_ar_dim1_2 = ",td_rv_ar_dim1_2);
  print("td_rv_ar_dim1_3 = ",td_rv_ar_dim1_3);
  print("td_rv_ar_dim2_1 = ",td_rv_ar_dim2_1);
  print("td_rv_ar_dim2_2 = ",td_rv_ar_dim2_2);
  print("td_m_ar_dim1_1 = ",td_m_ar_dim1_1);
  print("td_m_ar_dim1_2 = ",td_m_ar_dim1_2);
  print("td_m_ar_dim2_1 = ",td_m_ar_dim2_1);
  print("td_m_ar_dim2_2 = ",td_m_ar_dim2_2);
}
parameters {
} 
transformed parameters {
  vector[d] tp_v_ar_dim1_1[1] = { d_v1 };
  vector[d] tp_v_ar_dim1_2[2] = { d_v1, d_v2 };
  vector[d] tp_v_ar_dim1_3[1] = { d_v_ar[3] };
  vector[d] tp_v1 = d_m[,4];
  vector[d] tp_v_ar_dim1_4[1] = { tp_v1 };
  vector[d] tp_v_ar_dim1_5[2] = { d_v1, tp_v1 };
  vector[d] tp_v_ar_dim2_1[1,1] = { { d_v1 } };
  vector[d] tp_v_ar_dim2_2[2,3] = { { d_v1, d_v1, d_v1 }, { d_v2, d_v2, d_v2 } };

  row_vector[d] tp_rv_ar_dim1_1[1] = { d_rv1 };
  row_vector[d] tp_rv_ar_dim1_2[2] = { d_rv1, d_rv2 };
  row_vector[d] tp_rv_ar_dim1_3[1] = { d_rv_ar[3] };
  row_vector[d] tp_rv_ar_dim2_1[1,1] = { { d_rv1 } };
  row_vector[d] tp_rv_ar_dim2_2[2,3] = { { d_rv1, d_rv1, d_rv1 }, { d_rv2, d_rv2, d_rv2 } };

  matrix[d,d] tp_m_ar_dim1_1[1] = { d_m };
  matrix[d,d] tp_m_ar_dim1_2[2] = { d_m, d_m };
  matrix[d,d] tp_m_ar_dim2_1[1,1] = { { d_m } };
  matrix[d,d] tp_m_ar_dim2_2[2,3] = { { d_m, d_m, d_m }, { d_m, d_m, d_m } };

  print("tp_v_ar_dim1_1 = ",tp_v_ar_dim1_1);
  print("tp_v_ar_dim1_2 = ",tp_v_ar_dim1_2);
  print("tp_v_ar_dim1_3 = ",tp_v_ar_dim1_3);
  print("tp_v_ar_dim1_4 = ",tp_v_ar_dim1_4);
  print("tp_v_ar_dim1_5 = ",tp_v_ar_dim1_5);
  print("tp_v_ar_dim2_1 = ",tp_v_ar_dim2_1);
  print("tp_v_ar_dim2_2 = ",tp_v_ar_dim2_2);

  print("tp_rv_ar_dim1_1 = ",tp_rv_ar_dim1_1);
  print("tp_rv_ar_dim1_2 = ",tp_rv_ar_dim1_2);
  print("tp_rv_ar_dim1_3 = ",tp_rv_ar_dim1_3);
  print("tp_rv_ar_dim2_1 = ",tp_rv_ar_dim2_1);
  print("tp_rv_ar_dim2_2 = ",tp_rv_ar_dim2_2);
  print("tp_m_ar_dim1_1 = ",tp_m_ar_dim1_1);
  print("tp_m_ar_dim1_2 = ",tp_m_ar_dim1_2);
  print("tp_m_ar_dim2_1 = ",tp_m_ar_dim2_1);
  print("tp_m_ar_dim2_2 = ",tp_m_ar_dim2_2);
}
model {
}
generated quantities {
  vector[d] gq_v_ar_dim1_1[1] = { d_v1 };
  vector[d] gq_v_ar_dim1_2[2] = { d_v1, d_v2 };
  vector[d] gq_v_ar_dim1_3[1] = { d_v_ar[3] };
  vector[d] gq_v_ar_dim1_4[1] = { d_m[,3] };
  vector[d] gq_v_ar_dim2_1[1,1] = { { d_v1 } };
  vector[d] gq_v_ar_dim2_2[2,3] = { { d_v1, d_v1, d_v1 }, { d_v2, d_v2, d_v2 } };

  row_vector[d] gq_rv_ar_dim1_1[1] = { d_rv1 };
  row_vector[d] gq_rv_ar_dim1_2[2] = { d_rv1, d_rv2 };
  row_vector[d] gq_rv_ar_dim1_3[1] = { d_rv_ar[3] };
  row_vector[d] gq_rv_ar_dim2_1[1,1] = { { d_rv1 } };
  row_vector[d] gq_rv_ar_dim2_2[2,3] = { { d_rv1, d_rv1, d_rv1 }, { d_rv2, d_rv2, d_rv2 } };

  matrix[d,d] gq_m_ar_dim1_1[1] = { d_m };
  matrix[d,d] gq_m_ar_dim1_2[2] = { d_m, d_m };
  matrix[d,d] gq_m_ar_dim2_1[1,1] = { { d_m } };
  matrix[d,d] gq_m_ar_dim2_2[2,3] = { { d_m, d_m, d_m }, { d_m, d_m, d_m } };

  print("gq_v_ar_dim1_1 = ",gq_v_ar_dim1_1);
  print("gq_v_ar_dim1_2 = ",gq_v_ar_dim1_2);
  print("gq_v_ar_dim1_3 = ",gq_v_ar_dim1_3);
  print("gq_v_ar_dim1_4 = ",gq_v_ar_dim1_4);
  print("gq_v_ar_dim2_1 = ",gq_v_ar_dim2_1);
  print("gq_v_ar_dim2_2 = ",gq_v_ar_dim2_2);

  print("gq_rv_ar_dim1_1 = ",gq_rv_ar_dim1_1);
  print("gq_rv_ar_dim1_2 = ",gq_rv_ar_dim1_2);
  print("gq_rv_ar_dim1_3 = ",gq_rv_ar_dim1_3);
  print("gq_rv_ar_dim2_1 = ",gq_rv_ar_dim2_1);
  print("gq_rv_ar_dim2_2 = ",gq_rv_ar_dim2_2);
  print("gq_m_ar_dim1_1 = ",gq_m_ar_dim1_1);
  print("gq_m_ar_dim1_2 = ",gq_m_ar_dim1_2);
  print("gq_m_ar_dim2_1 = ",gq_m_ar_dim2_1);
  print("gq_m_ar_dim2_2 = ",gq_m_ar_dim2_2);
}
