functions {
  void foo_vec(vector a1) {
    matrix[2, 2] aa;
    matrix[3, 4] bb;
    aa += bb[1:2,1:2];
  }
}
data {
  int N;
  int J[N];
}
parameters {
  vector[N] b;
}
model {
  vector[N] mu = rep_vector(0.0, N);
  mu[J] += b[J];
}
