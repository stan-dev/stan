data {
  int N;
  int K;
  matrix[N,K] X;
}
parameters {
  vector[K] b;
}
generated quantities {
    vector[N] y_est = rep_vector(0,N);
    {
      y_est = X*b;
    }
}
