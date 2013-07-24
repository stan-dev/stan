# Stacks: robust regression and ridge regression 
# http://mathstat.helsinki.fi/openbugs/Examples/Stacks.html
# Model b) double exponential error

data {
  int<lower=0> N;
  int<lower=0> p;
  real Y[N];
  matrix[N,p] x;
} 

// to standardize the x's 
transformed data {
  matrix[N,p] z;
  row_vector[p] mean_x;
  real sd_x[p];
  for (j in 1:p) { 
    mean_x[j] <- mean(col(x,j)); 
    sd_x[j] <- sd(col(x,j)); 
    for (i in 1:N)  
      z[i,j] <- (x[i,j] - mean_x[j]) / sd_x[j]; 
  }
}

parameters {
  real beta0; 
  vector[p] beta;
  real<lower=0> sigmasq; 
} 

transformed parameters {
  real<lower=0> sigma;
  vector[N] mu;

  sigma <- sqrt(2) * sigmasq;
  mu <- beta0 + z * beta;
}

model {
  beta0 ~ normal(0, 316); 
  beta ~ normal(0, 316); 
  sigmasq ~ inv_gamma(.001, .001); 
  Y ~ double_exponential(mu, sigmasq);
} 

generated quantities {
  real b0;
  vector[p] b;
  real outlier_1;
  real outlier_3;
  real outlier_4;
  real outlier_21;

  for (j in 1:p)
    b[j] <- beta[j] / sd_x[j];
  b0 <- beta0 - mean_x * b;

  outlier_1  <- step(fabs((Y[1] - mu[1]) / sigma) - 2.5);
  outlier_3  <- step(fabs((Y[3] - mu[3]) / sigma) - 2.5);
  outlier_4  <- step(fabs((Y[4] - mu[4]) / sigma) - 2.5);
  outlier_21 <- step(fabs((Y[21] - mu[21]) / sigma) - 2.5);
}
