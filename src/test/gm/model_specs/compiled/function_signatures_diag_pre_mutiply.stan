data {
  matrix[3,3] m;
  vector[3] v;
  row_vector[3] rv;
}
transformed data {
  matrix[3,3] mt;
  mt <- diag_pre_multiply(v,m);  
  mt <- diag_pre_multiply(rv,m);
}
parameters {  
  matrix[3,3] pm;
  vector[3] pv;
  row_vector[3] prv;
}
transformed parameters {
  matrix[3,3] pmt;
  pmt <- diag_pre_multiply(pv,pm);  
  pmt <- diag_pre_multiply(prv,pm);
}
model {
  pv ~ normal(0,1);
}
