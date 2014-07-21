data {
  int<lower=0> N;
  int<lower=0> J;
  int<lower=0> K;
  int<lower=0> L;
  vector[N] y;
  vector[L] U;
  matrix[N,K] X;
  int county[N];
}
parameters {
  real<lower=0> sigma;
  vector[K] xi;
  vector[K] B_raw_temp;
  matrix[K,K] W;
  corr_matrix[K] Tau_b_raw;
  matrix[K,L] G_raw;
}
model {
  vector[N] y_hat;
  matrix[J,K] B;
  matrix[K,K] rho_b;
  matrix[K,K] Sigma_b_raw;
  matrix[K,L] G;
  matrix[J,K] B_raw_hat;
  matrix[J,K] B_raw;
  vector[K] sigma_b;

  xi ~ uniform(0, 100);

  Tau_b_raw ~ wishart(K+1, W);
  Sigma_b_raw <- inverse(Tau_b_raw);

  for (k in 1:K)
    for (j in 1:J)
      B[j,k] <- xi[k] * B_raw[j,k];

  for (k in 1:K){
    for (l in 1:L){
      G_raw[k,l] ~ normal(0, 100);
      G[k,l] <- xi[k] + G_raw[k,l];
    }
  }

  for (j in 1:J)
    for (k in 1:K)
      B_raw_hat[j,k] <- dot_product(row(G_raw,k),U);
  
  for (j in 1:J) {
    B_raw_temp ~ multi_normal(transpose(row(B_raw_hat,j)), Sigma_b_raw);    
    for (k in 1:K)
      B_raw[j,k] <- B_raw_temp[k];
  }

  for (k in 1:K) {
    sigma_b[k] <- fabs(xi[k]) * sqrt(Sigma_b_raw[k,k]);
    for (k_prime in 1:K)
      rho_b[k,k_prime] <- Sigma_b_raw[k,k_prime] 
        / sqrt(Sigma_b_raw[k,k] * Sigma_b_raw[k_prime,k_prime]);
  }
  
  for (i in 1:N)
    y_hat[i] <- dot_product(row(B,county[i]),row(X,i));

  y ~ normal(y_hat, sigma);
}
