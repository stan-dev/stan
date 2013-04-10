
# Schools: ranking school examination resutls using 
# multivariate hierarcical models 
#  http://www.openbugs.info/Examples/Schools.html

data {
  int<lower=0> N; 
  int<lower=0> M; 
  vector[N] LRT;
  int school[N]; 
  int School_denom[N, 3]; 
  int School_gender[N, 2]; 
  int VR[N, 2]; 
  real Y[N]; 
  int Gender[N]; 
  cov_matrix[3] R; 
} 

transformed data {
  vector[3] gamma_mu; 
  cov_matrix[3] gamma_Sigma; 
  cov_matrix[3] invR; 
  invR <- inverse(R); 
  gamma_mu[1] <- 0; 
  gamma_mu[2] <- 0; 
  gamma_mu[3] <- 0; 
  for (i in 1:3) for (j in 1:3) gamma_Sigma[i, j] <- 0; 
  for (i in 1:3) gamma_Sigma[i, i] <- 100; 
} 

parameters {
  real beta[8]; 
  vector[3] alpha[M]; 
  vector[3] gamma;
  cov_matrix[3] Sigma; 
  real theta; 
  real phi; 
} 

model {
  real Ymu[N]; 
  for(p in 1:N) {
    Ymu[p] <- alpha[school[p], 1] + alpha[school[p], 2] * LRT[p] + 
                   alpha[school[p], 3] * VR[p, 1] + beta[1] * LRT[p] * LRT[p] + 
                   beta[2] * VR[p, 2] + beta[3] * Gender[p] + 
                   beta[4] * School_gender[p, 1] + beta[5] * School_gender[p, 2] + 
                   beta[6] * School_denom[p, 1] + beta[7] * School_denom[p, 2] + 
                   beta[8] * School_denom[p, 3];
  }
  Y ~ normal(Ymu,  exp(-.5 * (theta + phi * LRT))); 
//  for(p in 1:N) {
//     Y[p] ~ normal(,  exp(-.5 * (theta + phi * LRT[p]))); 
//  }
  // min.var <- exp(-(theta + phi * (-34.6193))) # lowest LRT score = -34.6193
  // max.var <- exp(-(theta + phi * (37.3807)))  # highest LRT score = 37.3807

  # Priors for fixed effects:
  beta ~ normal(0, 100); 
  // for (k in 1:8)  beta[k] ~ normal(0.0, 100); 
  theta ~ normal(0.0, 100); 
  phi ~ normal(0.0, 100); 

  # Priors for random coefficients:
  for (m in 1:M) alpha[m] ~ multi_normal(gamma, Sigma); 
  # Hyper-priors:
  gamma ~ multi_normal(gamma_mu, gamma_Sigma); 
  Sigma ~ inv_wishart(3, invR); 
}

generated quantities {
  # real alpha1[M]; 
  real ranks[M]; 
  # for (m in 1:M)  alpha1[m] <- alpha[m, 1]; 
  ## compute ranks 
  for (j in 1:M) {
    real greater_than[M]; 
    for (k in 1:M) 
      greater_than[k] <- step(alpha[k, 1] - alpha[j, 1]); 
    ranks[j] <- sum(greater_than);
  }
} 
