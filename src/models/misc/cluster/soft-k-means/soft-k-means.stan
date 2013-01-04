data {
  int<lower=0> N;  // number of data points
  int<lower=1> D;  // number of dimensions
  int<lower=1> K;  // number of clusters
  vector[D] y[N];  // observations
}
transformed data {
  real<upper=0> neg_log_K;
  neg_log_K <- -log(K);
}
parameters {
  vector[D] mu[K]; // cluster means
}
transformed parameters {
  real<upper=0> soft_z[N,K]; // log unnormalized cluster assigns
  for (n in 1:N)
    for (k in 1:K)
      soft_z[n,k] <- neg_log_K - 0.5 * dot_self(mu[k] - y[n]);
}
model {
  for (k in 1:K)
    mu[k] ~ normal(0,1);  // prior
  for (n in 1:N)
    lp__ <- lp__ + log_sum_exp(soft_z[n]); // likelihood
}
