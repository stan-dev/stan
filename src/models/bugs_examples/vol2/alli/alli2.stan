# Alligators: multinomial - logistic regression 
#  http://www.openbugs.info/Examples/Aligators.html

## specify the model using Poisson distribution 

data {
  int I; // 4 
  int J; // 2 
  int K; // 5 
  int X[I, J, K];
} 

parameters {
  vector[K-1] alpha0;
  matrix[I-1, K-1] beta0;
  matrix[J-1, K-1] gamma0; 
  matrix[I, J] lambda;
}

transformed parameters {
  vector[K] alpha; 
  vector[K] beta[I];
  vector[K] gamma[J]; 

  alpha[1] <- 0;
  for (k in 1:(K-1))
    alpha[k+1] <- alpha0[k];

  for (i in 1:I)
    beta[i,1] <- 0;
  for (k in 1:K)
    beta[1,k] <- 0;

  for (i in 1:(I-1))
    for (k in 1:(K-1))
      beta[i+1,k+1] <- beta0[i,k];

  for (j in 1:J)
    gamma[j,1] <- 0;
  for (k in 1:K)
    gamma[1,k] <- 0;

  for (j in 1:(J-1))
    for (k in 1:(K-1))
      gamma[j+1,k+1] <- gamma0[j,k];

}

model {
  for (k in 2:K) { 
    alpha[k] ~ normal(0, 320);
    for (i in 2:I) 
      beta[i, k] ~ normal(0, 320);
    for (j in 2:J) 
      gamma[j, k] ~ normal(0, 320);
  } 

  # LIKELIHOOD  
  for (i in 1:I)  for (j in 1:J) {   
    lambda[i, j] ~ normal(0, 320);
    X[i, j] ~ poisson_log(lambda[i, j] + alpha + beta[i]  + gamma[j]);
  }
}

generated quantities {
  matrix[I, K] b;
  matrix[J, K] g;

  for (k in 1:K) { 
    for (i in 1:I) b[i,k] <- beta[i,k];
    for (j in 1:J) g[j,k] <- gamma[j,k];
  } 

  for (k in 1:K) {
    real mean_beta_k;
    mean_beta_k <- mean(col(b, k));
    for (i in 1:I) {
      b[i,k] <- beta[i,k] - mean_beta_k;
    }
  }

  for (k in 1:K) {
    real mean_gamma_k;
    mean_gamma_k <- mean(col(g, k));
    for (j in 1:J) {
      g[j,k] <- gamma[j,k] - mean_gamma_k;
    }
  }

}
