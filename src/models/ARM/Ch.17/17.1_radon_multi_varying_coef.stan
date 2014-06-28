data {
  int<lower=0> N;
  int<lower=0> J;
  vector[N] y;
  int<lower=0,upper=1> x[N];
  int county[N];
}
parameters {
  real<lower=0> sigma;
  vector<lower=0>[K] sigma_b;
  vector[K] mu_raw;
  vector[K] xi;
  matrix[J,K] B_raw;
  matrix[K,K] W;
  matrix[K,K] Tau_b_raw;
}
model {
  vector[N] y_hat;
  matrix[J,K] B;
  matrix[K,K] rho_b;
  matrix[K,K] Sigma_b_raw;
  vector[K] mu_raw;

  mu_raw ~ normal(0, 100);
  xi ~ uniform(0, 100);

  mu <- xi .* mu_raw;

  Tau_b_raw ~ wishart(W, K+1);
  Sigma_b_raw <- inverse(Tau_b_raw);

  for (j in 1:J) {
    for (k in 1:K)
      B[j,k] <- xi[k] * B_raw[j,k];
    B_raw[j,] ~ multi_normal(mu_raw, Sigma_b_raw);    
  }

  for (k in 1:K) {
    for (k_prime in 1:K)
      rho_b[k,k_prime] <- Sigma_b_raw[k,k_prime] 
        / sqrt(Sigma_b_raw[k,k] * Sigma_b_raw[k_prime,k_prime]);
    sigma_b[k] <- abs(xi[k]) * sqrt(Sigma_b_raw[k,k]);
  }
  
  for (i in 1:N)
    y_hat[i] <- dot_product(B[county[i],],X[i,]);

  y ~ normal(y_hat, sigma);
}
