# Endo: conditional inference in case-contrl studies 
# http://www.openbugs.info/Examples/Endo.html

# In this example, three methods of different 
# model specifications are used for one equivalent
# model. This is method 3. 


data {
  int n10; 
  int n01; 
  int n11; 
  int I;
  int J;
# int Y[I, 2]; 
# int est[I, 2]; 
} 

transformed data {
  int<lower=0> Y[I, 2]; 
  int<lower=0> est[I, 2]; 
  for (i in 1:I) {
    Y[i, 1] <- 1; 
    Y[i, 2] <- 0; 
  } 
  for (i in 1:n10) {
    est[i, 1] <- 1; 
    est[i, 2] <- 0; 
  } 
  for (i in (n10 + 1):(n10 + n01)) {  
    est[i, 1] <- 0;
    est[i, 2] <- 1;
  }
  for (i in (n10 + n01 + 1):(n10 + n01 + n11)) { 
    est[i, 1] <- 1;  
    est[i, 2] <- 1;
  }
  for (i in (n10 + n01 + n11 + 1):I ) { 
    est[i, 1] <- 0;  
    est[i, 2] <- 0;
  }
} 

parameters {
  real beta; 
  real beta0[I];
} 

model {
  # METHOD 3 fit standard Poisson regressions relative to baseline
  beta ~ normal(0, 1000); 
  beta0 ~ normal(0, 1000);
  for (i in 1:I) 
    for (j in 1:J)
      Y[i, j] ~ poisson_log(beta0[i] + beta * est[i, j]); 
} 
