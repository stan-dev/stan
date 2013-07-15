data {
  int<lower=0> N; 
  vector[N] kid_score;
  int mom_work[N];
}
 
transformed data {
  matrix[N,4] X; 
  int mom_val;
  for (i in 1:N) {
  mom_val <- mom_work[i];
    for (j in 1:mom_val)
      X[i,j] <- 1;
  }
}

parameters {
  vector[4] beta;
  real<lower=0> sigma;
} 

model {
  kid_score ~ normal(X * beta, sigma);
}
