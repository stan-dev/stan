functions {
  vector lr_lp(vector beta, vector theta, real[] x, int[] y) {
    real lp = bernoulli_logit_lpmf(y | beta[1] + to_vector(x) * beta[2]);
    return [lp]';
  }
}
data {
  // N = 12 data points
  int y[12];
  real x[12];
}
transformed data {
  // K = 3 shards
  int ys[3, 4] = { y[1:4], y[5:8], y[9:12] };
  real xs[3, 4] = { x[1:4], x[5:8], x[9:12] };
  vector[0] theta[3];
}
parameters {
  vector[2] beta;
}
model {
  target += sum(map_rect(lr_lp, beta, theta, xs, ys));
}
