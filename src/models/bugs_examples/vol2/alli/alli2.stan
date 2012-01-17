# Alligators: multinomial - logistic regression 
#  http://www.openbugs.info/Examples/Aligators.html

## specify the model using Poisson distribution 

## status (works)

data {
  int I; // 4 
  int J; // 2 
  int K; // 5 
  int X[I, J, K];
} 

parameters {
  double alpha[K - 1]; 
  double beta[I - 1, K - 1]; 
  double gamma[J - 1, K - 1]; 
  double lambda[I, J]; 
}

transformed parameters {
  double yaalpha[K]; 
  double yabeta[I, K];
  double yagamma[J, K];
  yaalpha[1] <- 0; 
  for (k in 2:K)  yaalpha[k] <- alpha[k - 1];
  for (k in 1:K) {
    yabeta[1, k] <- 0;
    yagamma[1, k] <- 0;
  }
  for (i in 2:I) yabeta[i, 1] <- 0;
  for (j in 2:J) yagamma[j, 1] <- 0;
  for (k in 2:K) {
    for (i in 2:I)  yabeta[i, k] <- beta[i - 1, k - 1];
    for (j in 2:J)  yagamma[j, k] <- gamma[j - 1, k - 1];
  }
} 

model {
  for (k in 1:(K - 1)) { 
    alpha[k] ~ normal(0, 320);
    for (i in 1:(I - 1)) beta[i, k] ~ normal(0, 320);
    for (i in 1:(J - 1)) gamma[i, k] ~ normal(0, 320);
  } 

  # LIKELIHOOD    
  for (i in 1:I) {   
    for (j in 1:J) {   
      lambda[i, j] ~ normal(0, 320);
      for (k in 1:K)       
        X[i, j, k] ~ poisson(exp(lambda[i, j] + yaalpha[k] + yabeta[i, k]  + yagamma[j, k]));
    }  
  }
}

