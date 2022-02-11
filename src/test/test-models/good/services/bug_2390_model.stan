transformed data {
  int N = 5;
  int K = 1;
  array[5] int<lower=0, upper=1> y = {1, 1, 1, 1, 0};
  matrix[5, 1] X = [[1], [1], [1], [1], [1]];
}
parameters {
  vector[K] b;
}
model {
  y ~ bernoulli_logit(X * b);
}

