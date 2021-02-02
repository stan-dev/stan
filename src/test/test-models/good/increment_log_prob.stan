transformed data {
  int n;
  array[2] int nn;
  array[3, 4] int nnn;
  real x;
  array[5] real xx;
  array[6, 7] real xxx;
  array[8, 9, 10] real xxxx;
  vector[2] v;
  array[4] vector[3] vv;
  array[5, 6] vector[4] vvv;
  row_vector[2] rv;
  array[4] row_vector[3] rvv;
  array[5, 6] row_vector[4] rvvv;
  matrix[7, 8] m;
  array[2] matrix[7, 8] mm;
  array[3, 4] matrix[7, 8] mmm;
}
parameters {
  real p_x;
  array[5] real p_xx;
  array[6, 7] real p_xxx;
  array[8, 9, 10] real p_xxxx;
  vector[2] p_v;
  array[4] vector[3] p_vv;
  array[5, 6] vector[4] p_vvv;
  row_vector[2] p_rv;
  array[4] row_vector[3] p_rvv;
  array[5, 6] row_vector[4] p_rvvv;
  matrix[7, 8] p_m;
  array[2] matrix[7, 8] p_mm;
  array[3, 4] matrix[7, 8] p_mmm;
}
model {
  target += n;
  target += nn;
  target += nnn;
  target += x;
  target += xx;
  target += xxx;
  target += xxxx;
  target += v;
  target += vv;
  target += vvv;
  target += rv;
  target += rvv;
  target += rvvv;
  target += m;
  target += mm;
  target += mmm;
  target += p_x;
  target += p_xx;
  target += p_xxx;
  target += p_xxxx;
  target += p_v;
  target += p_vv;
  target += p_vvv;
  target += p_rv;
  target += p_rvv;
  target += p_rvvv;
  target += p_m;
  target += p_mm;
  target += p_mmm;
}

