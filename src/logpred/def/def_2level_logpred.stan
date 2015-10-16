data {
  int<lower=0> S;      // number of samples from posterior

  int<lower=0> N;      // number of data points
  int<lower=0> I;      // number of count types
  int<lower=0> x_heldout[N,I]; // x[n,i]: count of type i for observation n

  int<lower=1> K_2; // number of latent variables in layer 2
  int<lower=1> K_1; // number of latent variables in layer 1

  matrix<lower=0>[K_1, I] W_0[S];
  matrix<lower=0>[N, K_1] z_1[S];
}

model { }

generated quantities {
  real log_pred;
  real ave_log_pred;
  real inner_product;

  ave_log_pred <- 0.0;
  for (n in 1:N) {
    for (i in 1:I) {
      log_pred <- 0.0;
      for (s in 1:S) {
        inner_product <- 0.0;
        for (k in 1:K_1) {
          inner_product <- inner_product + z_1[s,n,k] * W_0[s,k,i];
        }
        log_pred <- log_pred + poisson_log(x_heldout[n,i], inner_product);
      }
      log_pred <- log_pred / S;
      ave_log_pred <- ave_log_pred + log_pred;
    }
  }
  ave_log_pred <- ave_log_pred / (N*I);
}
