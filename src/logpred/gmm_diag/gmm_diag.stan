data {
  int<lower=0> K;       // number of mixture components
  int<lower=0> N;       // number of data points
  int<lower=0> D;       // dimension
  vector[D] y[N];       // observations

  real<lower=0> alpha0; // dirichlet prior
}

transformed data {
  vector<lower=0>[K] alpha0_vec;
  for (k in 1:K) {
    alpha0_vec[k] <- alpha0;
  }
}

parameters {
  simplex[K] theta;             // mixing proportions
  vector[D] mu[K];              // locations of mixture components
  vector<lower=0>[D] sigma[K];  // standard deviations of mixture components
}

model {
  real ps[K];

  // prior
  theta ~ dirichlet(alpha0_vec);
  for (k in 1:K) {
      mu[k] ~ normal(0.0, 10.0);
      sigma[k] ~ normal(1.0, 1.0);
  }

  // likelihood
  for (n in 1:N) {
    for (k in 1:K) {
      ps[k] <- log(theta[k]) + normal_log(y[n], mu[k], sigma[k]);
    }
    increment_log_prob(log_sum_exp(ps));
  }
}
