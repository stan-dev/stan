functions {
  vector lr_rng(vector beta, vector theta, real[] x, int[] y) {
    int draw = bernoulli_rng(inv_logit(sum(beta[1] + to_vector(x) * beta[2])));
    return [ draw ]';
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
transformed parameters {
  vector[12] bar = map_rect(lr_rng, beta, theta, xs, ys);
}
