functions {
  matrix foo(matrix x) {
    matrix[2,2] lf0;
    matrix[3,3] lf1 = x;
    return lf1;
  }
}
data {
  matrix[3,3] d1;
  matrix[3,3] d_a1[10];
}
transformed data {
  matrix[3,3] td1 = d1;
  matrix[3,3] td_a1[10] = d_a1;
  row_vector[3] td2 = td1[1];
  {
    matrix[3,3] loc_td1 = d1;
    row_vector[3] loc_td2 = td1[1];
  }
}
parameters {
  matrix[3,3] par1;
}
transformed parameters {
  matrix[3,3] tp1 = par1;
  matrix[3,3] tp2 = d1;
  {
    matrix[3,3] loc_tp1 = par1;
    matrix[3,3] loc_tp2 = d1;
  }
}
model {
  matrix[3,3] lm1 = d1;
  matrix[3,3] lm2 = lm1;
}
generated quantities {
  matrix[3,3] gq1 = d1;
  row_vector[3] gq2 = gq1[1];
  {
    matrix[3,3] lgq1 = d1;
  }
}
