data {
  int<lower=0> K;                 // number of mixture components
  int<lower=0> N;                 // number of data points
  int<lower=0> D;                 // dimension
  vector[D] y_heldout[N];         // observations

  int<lower=0> S;                 // number of samples from posterior

  simplex[K] theta[S];            // mixing proportions
  vector[D] mu[K,S];              // locations of mixture components
  vector<lower=0>[D] sigma[K,S];  // stanadard deviations of mixture components
}

model { }

generated quantities {
  real ps[K];
  real log_pred;
  real ave_log_pred;
  ave_log_pred <- 0.0;
  for (n in 1:N) {
    log_pred <- 0.0;
    for (s in 1:S) {
      for (k in 1:K) {
        ps[k] <- log(theta[s,k]) + normal_log(y_heldout[n], mu[k,s], sigma[k,s]);
      }
      log_pred <- log_pred + log_sum_exp(ps);
    }
    log_pred <- log_pred / S;
    ave_log_pred <- ave_log_pred + log_pred;
  }
  ave_log_pred <- ave_log_pred / N;
}
