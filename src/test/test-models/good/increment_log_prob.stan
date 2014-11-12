transformed data {
  int n;
  int nn[2];
  int nnn[3,4];
  
  real x;
  real xx[5];
  real xxx[6,7];
  real xxxx[8,9,10];

  vector[2] v;
  vector[3] vv[4];
  vector[4] vvv[5,6];

  row_vector[2] rv;
  row_vector[3] rvv[4];
  row_vector[4] rvvv[5,6];

  matrix[7,8] m;
  matrix[7,8] mm[2];
  matrix[7,8] mmm[3,4];
}
parameters {
  real p_x;
  real p_xx[5];
  real p_xxx[6,7];
  real p_xxxx[8,9,10];

  vector[2] p_v;
  vector[3] p_vv[4];
  vector[4] p_vvv[5,6];

  row_vector[2] p_rv;
  row_vector[3] p_rvv[4];
  row_vector[4] p_rvvv[5,6];

  matrix[7,8] p_m;
  matrix[7,8] p_mm[2];
  matrix[7,8] p_mmm[3,4];
}
model {
  increment_log_prob(n);
  increment_log_prob(nn);
  increment_log_prob(nnn);

  increment_log_prob(x);
  increment_log_prob(xx);
  increment_log_prob(xxx);
  increment_log_prob(xxxx);

  increment_log_prob(v);
  increment_log_prob(vv);
  increment_log_prob(vvv);

  increment_log_prob(rv);
  increment_log_prob(rvv);
  increment_log_prob(rvvv);

  increment_log_prob(m);
  increment_log_prob(mm);
  increment_log_prob(mmm);


  increment_log_prob(p_x);
  increment_log_prob(p_xx);
  increment_log_prob(p_xxx);
  increment_log_prob(p_xxxx);

  increment_log_prob(p_v);
  increment_log_prob(p_vv);
  increment_log_prob(p_vvv);

  increment_log_prob(p_rv);
  increment_log_prob(p_rvv);
  increment_log_prob(p_rvvv);

  increment_log_prob(p_m);
  increment_log_prob(p_mm);
  increment_log_prob(p_mmm);

}
