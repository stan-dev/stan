data {
  matrix[3, 4] a_d;
}
transformed data {
  array[5] int u;
  u = csr_extract_u(a_d);
}
parameters {
  real y;
  matrix[3, 4] a_p;
}
transformed parameters {
  real v;
  {
    array[5] int u_p;
    u_p = csr_extract_u(a_d);
    u_p = csr_extract_u(a_p);
    v = 3;
  }
}
model {
  y ~ normal(0, 1);
}

