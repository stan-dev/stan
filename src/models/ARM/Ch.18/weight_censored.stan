data {
  int<lower=0> N;
  int<lower=0> N_obs;
  int<lower=0> N_cens;
  vector[N] weight;
  vector[N] height;
  real<lower=max(weight)> C;
}
transformed data {
  vector[N] c_height;
  vector[N_obs] weight_obs;
  vector[N_obs] c_height_obs;
  vector[N_cens] c_height_cens;
  int i;
  int j;
  c_height <- height - mean(height);
  i <- 1;
  j <- 1;
  for (n in 1:N) {
    if (weight[n] == C) {
      c_height_cens[i] <- c_height[n];
      i <- i + 1;
    } else {
      weight_obs[j] <- weight[n];
      c_height_obs[j] <- c_height[n];
      j <- j + 1;
    }
  }
}
parameters {
  vector<lower=C>[N_cens] weight_cens;
  real a;
  real b;
  real<lower=0> sigma;
}
model {
  weight_obs ~ normal(a + b * c_height_obs, sigma);
  weight_cens ~ normal(a + b * c_height_cens, sigma);
}
