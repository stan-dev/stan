data {
  int<lower=1> K;  // num categories
  int<lower=1> V;  // num words
  int<lower=0> N;  // num supervised items
  int<lower=1> M;  // num unsupervised items
  int<lower=1,upper=V> u[M]; // unsup words
  int<lower=1,upper=V> w[N]; // words
  int<lower=1,upper=K> z[N]; // categories
  vector<lower=0>[K] alpha;  // transit prior
  vector<lower=0>[V] beta;   // emit prior
}
parameters {
  simplex[K] theta[K];  // transit probs
  simplex[V] phi[K];    // emit probs
}
model {
  for (k in 1:K) 
    theta[k] ~ dirichlet(alpha);
  for (k in 1:K)
    phi[k] ~ dirichlet(beta);
  for (n in 1:N)
    w[n] ~ categorical(phi[z[n]]);
  for (n in 2:N)
    z[n] ~ categorical(theta[z[n-1]]);

  // forward algorithm
  { 
    real acc[K];
    real gamma[M,K];
    for (k in 1:K)
      gamma[1,k] <- log(phi[k,u[1]]);
    for (m in 2:M) {
      for (k in 1:K) {
        for (j in 1:K)
          acc[j] <- log(theta[j,k]) + log(phi[k,u[m]]);
        gamma[m,k] <- log_sum_exp(acc);
      }
    }
    lp__ <- lp__ + log_sum_exp(gamma[M]);
  }
}
generated quantities {
  int<lower=1,upper=K> y_star[M];
  real log_p_y_star;

  // Viterbi algorithm
  { 
    int back_ptr[M,K];
    real best_logp[M,K];
    real best_total_logp;
    for (k in 1:K)
      best_logp[1,K] <- log(phi[k,u[1]]);
    for (m in 2:M) {
      for (k in 1:K) {
        best_logp[m,k] <- negative_infinity();
        for (j in 1:K) {
          real logp;
          logp <- best_logp[m-1,j] + log(theta[j,k]) + log(phi[k,u[m]]);
          if (logp > best_logp[m,k]) {
            back_ptr[m,k] <- j;
            best_logp[m,k] <- logp;
          }
        }
      }
    }
    log_p_y_star <- max(best_logp[M]);
    for (k in 1:K)
      if (best_logp[M,k] == log_p_y_star)
        y_star[M] <- k;
    for (m in 2:(M-1))
      y_star[M-m] <- back_ptr[M-m+1,y_star[M-m+1]];
  }
}
