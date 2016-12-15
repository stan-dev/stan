functions {
  void foo1() {
    vector[2] lf1;
    vector[2] lf2 = lf1;
    print("i am void");
  }
  vector foo2(int x) {
    vector[x] lf1;
    return lf1;
  }
}
data {
  int d;
  vector[d] d_v;
  row_vector[d] d_rv;
  vector[d] d_v_ar[d];
  row_vector[d] d_rv_ar[d];
  matrix[d, d] d_m;
}
transformed data {
  vector[d] td_v1 = d_v;
  vector[d] td_v2 = d_v_ar[1];
  vector[d] td_v3 = d_m[, 1];
  vector[d] td_v_ar[d] = d_v_ar;

  row_vector[d] td_rv1 = d_rv;
  row_vector[d] td_rv2 = d_rv_ar[1];
  row_vector[d] td_rv3 = d_m[2];
  row_vector[d] td_rv_ar[d] = d_rv_ar;

  print("td_v1 = ", td_v1);
  print("td_v2 = ", td_v2);
  print("td_v3 = ", td_v3);
  print("td_v_ar = ", td_v_ar);
  print("td_rv1 = ", td_rv1);
  print("td_rv2 = ", td_rv2);
  print("td_rv3 = ", td_rv3);
  print("td_rv_ar = ", td_rv_ar);

  {
    vector[d] local_v1 = d_v;
    vector[d] local_v2 = d_v_ar[1];
    vector[d] local_v3 = d_m[, 1];
    vector[d] local_v_ar[d] = d_v_ar;

    row_vector[d] local_rv1 = d_rv;
    row_vector[d] local_rv2 = d_rv_ar[1];
    row_vector[d] local_rv3 = d_m[2];
    row_vector[d] local_rv_ar[d] = d_rv_ar;

    print("local_v1 = ", local_v1);
    print("local_v2 = ", local_v2);
    print("local_v3 = ", local_v3);
    print("local_v_ar = ", local_v_ar);
    print("local_rv1 = ", local_rv1);
    print("local_rv2 = ", local_rv2);
    print("local_rv3 = ", local_rv3);
    print("local_rv_ar = ", local_rv_ar);
  }
  foo1();
}
parameters {
  matrix<lower = 0, upper = 1>[d, d] p_m;
}
transformed parameters {
  vector[d] tp_v1 = d_v;
  vector[d] tp_v2 = d_v_ar[1];
  vector[d] tp_v3 = d_m[, 1];
  vector[d] tp_v_ar4[d] = d_v_ar;

  vector[d] tp_v5 = tp_v1;
  vector[d] tp_v6 = tp_v_ar4[1];
  vector[d] tp_v7 = p_m[, 1];
  vector[d] tp_v_ar8[d] = tp_v_ar4;
  
  row_vector[d] tp_rv1 = d_rv;
  row_vector[d] tp_rv2 = d_rv_ar[1];
  row_vector[d] tp_rv3 = d_m[2];
  row_vector[d] tp_rv_ar4[d] = d_rv_ar;

  row_vector[d] tp_rv5 = tp_rv1;
  row_vector[d] tp_rv6 = tp_rv_ar4[1];
  row_vector[d] tp_rv7 = p_m[2, ];
  row_vector[d] tp_rv_ar8[d] = tp_rv_ar4;

  print("tp_v1 = ", tp_v1);
  print("tp_v2 = ", tp_v2);
  print("tp_v3 = ", tp_v3);
  print("tp_v_ar4 = ", tp_v_ar4);
  print("tp_v5 = ", tp_v5);
  print("tp_v6 = ", tp_v6);
  print("tp_v7 = ", tp_v7);
  print("tp_v_ar8 = ", tp_v_ar8);


  print("tp_rv1 = ", tp_rv1);
  print("tp_rv2 = ", tp_rv2);
  print("tp_rv3 = ", tp_rv3);
  print("tp_rv_ar = ", tp_rv_ar4);
  print("tp_rv5 = ", tp_rv5);
  print("tp_rv6 = ", tp_rv6);
  print("tp_rv7 = ", tp_rv7);
  print("tp_rv_ar8 = ", tp_rv_ar8);

  {
    vector[d] local_v1 = d_v;
    vector[d] local_v2 = d_v_ar[1];
    vector[d] local_v3 = d_m[, 1];
    vector[d] local_v_ar4[d] = d_v_ar;
    vector[d] local_v5 = tp_v1;
    vector[d] local_v6 = tp_v_ar4[1];
    vector[d] local_v7 = p_m[, 1];
    vector[d] local_v_ar8[d] = tp_v_ar4;

    row_vector[d] local_rv1 = d_rv;
    row_vector[d] local_rv2 = d_rv_ar[1];
    row_vector[d] local_rv3 = d_m[2];
    row_vector[d] local_rv_ar4[d] = d_rv_ar;
    row_vector[d] local_rv5 = tp_rv1;
    row_vector[d] local_rv6 = tp_rv_ar4[1];
    row_vector[d] local_rv7 = p_m[1];
    row_vector[d] local_rv_ar8[d] = tp_rv_ar4;
    
    print("local_v1 = ", local_v1);
    print("local_v2 = ", local_v2);
    print("local_v3 = ", local_v3);
    print("local_v_ar4 = ", local_v_ar4);
    print("local_v5 = ", local_v5);
    print("local_v6 = ", local_v6);
    print("local_v7 = ", local_v7);
    print("local_v_ar8 = ", local_v_ar8);
    print("local_rv1 = ", local_rv1);
    print("local_rv2 = ", local_rv2);
    print("local_rv3 = ", local_rv3);
    print("local_rv_ar4 = ", local_rv_ar4);
    print("local_rv5 = ", local_rv5);
    print("local_rv6 = ", local_rv6);
    print("local_rv7 = ", local_rv7);
    print("local_rv_ar8 = ", local_rv_ar8);
  }
}
model {
}
generated quantities {
}
