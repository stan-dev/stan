data {
  matrix[3, 4] a_d;
}
transformed data {
  array[5] int v_d;
  v_d = csr_extract_v(a_d);
}
parameters {
  real y_p;
  matrix[3, 4] a_p;
}
transformed parameters {
  real v;
  {
    array[5] int v_p;
    v_p = csr_extract_v(a_d);
    v_p = csr_extract_v(a_p);
    v = 3;
  }
}
model {
  y_p ~ normal(0, 1);
}

