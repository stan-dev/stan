data {
  int<lower=0> N;
  int<lower=0> J;
  int<lower=0> K;
  vector[N] y;
  matrix[N,K] X;
  int county[N];
  matrix[K,K] W;
}
parameters {
  real<lower=0> sigma;
  vector[K] mu_raw;
  vector[K] xi;
  corr_matrix[K] Tau_b_raw;
  vector[K] B_raw_temp;
}
model {
  vector[N] y_hat;
  matrix[J,K] B;
  matrix[K,K] rho_b;
  matrix[K,K] Sigma_b_raw;
  vector[K] mu;
  matrix[J,K] B_raw;
  vector[K] sigma_b;

  mu_raw ~ normal(0, 100);
  xi ~ uniform(0, 100);

  mu <- xi .* mu_raw;

  Tau_b_raw ~ wishart(K+1, W);
  Sigma_b_raw <- inverse(Tau_b_raw);

  for (j in 1:J) {
    B_raw_temp ~ multi_normal(mu_raw, Sigma_b_raw);    
    for (k in 1:K) {
      B_raw[j,k] <- B_raw_temp[k];
      B[j,k] <- xi[k] * B_raw[j,k];
    }
  }

  for (k in 1:K) {
    for (k_prime in 1:K)
      rho_b[k,k_prime] <- Sigma_b_raw[k,k_prime] 
        / sqrt(Sigma_b_raw[k,k] * Sigma_b_raw[k_prime,k_prime]);
    sigma_b[k] <- fabs(xi[k]) * sqrt(Sigma_b_raw[k,k]);
  }
  
  for (i in 1:N)
    y_hat[i] <- dot_product(row(B,county[i]),row(X,i));

  y ~ normal(y_hat, sigma);
}
