functions {
  void foo_vec(vector a1) {
    matrix[2, 2] aa;
    matrix[3, 4] bb;
    aa -= bb[1:2,1:2];
  }
}
data {
  int N;
  int J[N];
}
parameters {
  vector[N] p_v1;
}
model {
  vector[N] m_v1 = rep_vector(1.0, N);
  m_v1[J] -= p_v1[J];
}
generated quantities {
  matrix[N,N] gq_m1;
  row_vector[N] gq_rv1 = rep_row_vector(1.0, N);
  vector[N] gq_v1 = rep_vector(1.0, N);
  gq_m1[J,1] -= gq_v1[J];
  gq_m1[1,J] -= gq_rv1[J];
}
