# Bones: latent trait model for multiple ordered 
# categorical responses 
## http://www.openbugs.info/Examples/Bones.html


## Note: 
## 1. Since it is just the response that is 
##    modelled as categorical distribution,
##    we should be able to run the model now except handling 
##    the missing. However, the data structure is a bit 
##    difficult to deal with (Allowing some redundancy 
##    in the transformed parameters (Q here), the model
##    is fine in Stan. 
## 2. The missing data is recoded as `-1`, which is 
##    not modeled for `gamma` as in the OpenBUGS example
##    and not modeled for `grade`. 


data {
  int<lower=0> nChild; 
  int<lower=0> nInd; 
  real gamma[nInd, 4];  // -1 indicates NA in original R dump data (bones.data.R.0)
  real delta[nInd]; 
  int<lower=0> ncat[nInd]; 
  int grade[nChild, nInd];  // -1 indicates NA in original R dump data (bones.data.R.0)
} 


parameters {
  real theta[nChild]; 
}

model { 
  real p[nChild, nInd, 5];
  real Q[nChild, nInd, 4];
  theta ~ normal(0.0, 36); 
  for(i in 1:nChild) {
    # Probability of observing grade k given theta
    for (j in 1:nInd) {
      # Cumulative probability of > grade k given theta
      for (k in 1:(ncat[j] - 1)) { 
        Q[i, j, k] <- inv_logit(delta[j] * (theta[i] - gamma[j, k])); 
      } 

      p[i, j, 1] <- 1 - Q[i, j, 1];
      for (k in 2:(ncat[j] - 1))  
        p[i, j, k] <- Q[i, j, k - 1] - Q[i, j, k];
      p[i, j, ncat[j]] <- Q[i, j, ncat[j] - 1];
      // grade[i, j] - 1 ~ categorical(p[i, j, 1:ncat[j]])
      
      // We use lp__ instead, since grade[i, j] has categorical distribution
      // with varying dimension. 
      // If grade[i, j] = -1, it is missing, no contribution for lp__ then. 
      if (grade[i, j] != -1) {
        lp__ <- lp__ + log(p[i, j, grade[i, j]]);  
      } 
    }
  }
}



