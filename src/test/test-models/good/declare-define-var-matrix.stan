functions {
  matrix foo(int a) {
    matrix[a,a] lf0;
    matrix[a,a] lf1 = lf0;
    return lf1;
  }
}
data {
  int d;
  matrix [d,d] d_m;
  matrix [d,d] d_m_ar[d];
  
}
transformed data {
  matrix[d,d] td_m1 = d_m;
  matrix[d,d] td_m2 = d_m_ar[1];
  matrix[d,d] td_m_ar3[d] = d_m_ar;

  print("td_m1 = ", td_m1);
  print("td_m2 = ", td_m2);
  print("td_m_ar3 = ", td_m_ar3);
}
transformed parameters {
  matrix[d,d] tp_m1 = d_m;
  matrix[d,d] tp_m2 = d_m_ar[1];
  matrix[d,d] tp_m_ar3[d] = d_m_ar;
  matrix[d,d] tp_m4 = tp_m1;
  matrix[d,d] tp_m5 = tp_m_ar3[1];
  matrix[d,d] tp_m_ar6[d] = tp_m_ar3;

  print("tp_m1 = ", tp_m1);
  print("tp_m2 = ", tp_m2);
  print("tp_m_ar3 = ", tp_m_ar3);
  print("tp_m4 = ", tp_m4);
  print("tp_m5 = ", tp_m5);
  print("tp_m_ar6 = ", tp_m_ar6);
  {
    matrix[d,d] local_m1 = d_m;
    matrix[d,d] local_m2 = d_m_ar[1];
    matrix[d,d] local_m_ar3[d] = d_m_ar;
    matrix[d,d] local_m4 = tp_m1;
    matrix[d,d] local_m5 = tp_m_ar3[1];
    matrix[d,d] local_m_ar6[d] = tp_m_ar3;
    print("local_m1 = ", local_m1);
    print("local_m2 = ", local_m2);
    print("local_m_ar3 = ", local_m_ar3);
    print("local_m4 = ", local_m4);
    print("local_m5 = ", local_m5);
    print("local_m_ar6 = ", local_m_ar6);
  }
}
model {
}
generated quantities {
  matrix[d,d] gq_m1 = d_m;
  matrix[d,d] gq_m2 = d_m_ar[1];
  matrix[d,d] gq_m_ar3[d] = d_m_ar;
  matrix[d,d] gq_m4 = tp_m1;
  matrix[d,d] gq_m5 = tp_m_ar3[1];
  matrix[d,d] gq_m_ar6[d] = tp_m_ar3;

  print("gq_m1 = ", gq_m1);
  print("gq_m2 = ", gq_m2);
  print("gq_m_ar3 = ", gq_m_ar3);
  print("gq_m4 = ", gq_m4);
  print("gq_m5 = ", gq_m5);
  print("gq_m_ar6 = ", gq_m_ar6);  
}
