# Endo: conditional inference in case-contrl studies 
# http://www.openbugs.info/Examples/Endo.html

# In this example, three methods of different 
# model specifications are used for one equivalent
# model. This is method 1. 

data {
  int n10; 
  int n01; 
  int n11; 
  int I;
} 

transformed data {
  int J;
  int Y[2, I]; 
  vector<lower=0>[I] est[2];
  vector[I] est1m2;
  J <- 2;
  for (i in 1:I) {
    Y[1, i] <- 1; 
    Y[2, i] <- 0; 
  } 
  for (i in 1:n10) {
    est[1, i] <- 1; 
    est[2, i] <- 0; 
  } 
  for (i in (n10 + 1):(n10 + n01)) {  
    est[1, i] <- 0;
    est[2, i] <- 1;
  }
  for (i in (n10 + n01 + 1):(n10 + n01 + n11)) { 
    est[1, i] <- 1;  
    est[2, i] <- 1;
  }
  for (i in (n10 + n01 + n11 + 1):I ) { 
    est[1, i] <- 0;  
    est[2, i] <- 0;
  }
  est1m2 <- est[1] - est[2];
} 

parameters {
  real beta; 
} 

model {
  # METHOD 1: Logistic regression 
  beta ~ normal(0, 1000); 
  Y[1] ~ binomial_logit(1, beta * est1m2);
} 
