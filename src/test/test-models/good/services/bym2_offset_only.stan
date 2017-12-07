data {
  int<lower=0> N;
  int<lower=0> N_edges;
  int<lower=1, upper=N> node1[N_edges];  // node1[i] adjacent to node2[i]
  int<lower=1, upper=N> node2[N_edges];  // and node1[i] < node2[i]

  int<lower=0> y[N];              // count outcomes
  vector<lower=0>[N] E;           // exposure

  real<lower=0> scaling_factor; //the scaling factor to make the ICAR variances approxiamtely one
}
transformed data {
  vector[N] log_E = log(E);
}
parameters {
  real beta0;                // intercept

  real<lower=0> sigma;        // overall standard deviation
  real<lower=0, upper=1> rho; // proportion unstructured vs. spatially structured variance

  vector[N] theta;       // heterogeneous effects
  vector[N - 1] phi_raw; // raw spatial effects
}
transformed parameters {
  vector[N] phi;
  vector[N] convolved_re;

  phi[1:(N - 1)] = phi_raw;
  phi[N] = -sum(phi_raw);

  // NB: scaling_factor scales the spatial effect so the variance is approxiamtely 1.
  // This is NOT a magic number, and comes as data.
  // Divide by sqrt of scaling factor to properly scale precision matrix phi.
  convolved_re =  sqrt(1 - rho) * theta + sqrt(rho / scaling_factor) * phi;
}
model {
  y ~ poisson_log(log_E + beta0 + convolved_re * sigma);

  // This is the prior for phi! (up to proportionality)
  target += -0.5 * dot_self(phi[node1] - phi[node2]);

  beta0 ~ normal(0.0, 2.5);
  theta ~ normal(0.0, 1.0);
  sigma ~ normal(0,5);
  rho ~ beta(0.5, 0.5);
}
generated quantities {
  real log_precision = -2.0 * log(sigma);
  real logit_rho = log(rho / (1.0 - rho));
  vector[N] eta = log_E + beta0 + convolved_re * sigma;
  vector[N] mu = exp(eta);
  vector[N] log_lik;
  int y_rep[N];
  if (max(eta) > 20) {
    print("max eta too big: ", max(eta));
    for (n in 1:N) {
      y_rep[n] = -1;
      log_lik[n] = not_a_number();
    }
  } else {
      for (n in 1:N) {
        y_rep[n] = poisson_log_rng(eta[n]);
        log_lik[n] = poisson_log_lpmf(y[n] | eta[n]);
      }
  }
}

