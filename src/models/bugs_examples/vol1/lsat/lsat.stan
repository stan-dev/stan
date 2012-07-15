# LSAT: item response
# http://www.openbugs.info/Examples/Lsat.html

data {
  int(0,) N; // 1000, number of students
  int(0,) R; // 32, number of patterns of results: 2^T
  int(0,) T; // 5, number of questions
  int(0,) culm[R];
  int(0,) response[R,T];
}

transformed data {
  int r[N,T];
  
  for (j in 1:culm[1]) {
    for (k in 1:T) {
      r[j,k] <- response[1,k];
   } 
  }
  for (i in 2:R) {
    for (j in (culm[i-1] + 1):culm[i]) {
      for (k in 1:T) {
        r[j,k] <- response[i,k];
      }
    }
  } 
}
parameters {
  real alpha[T];
  real theta[N];
  real(0,) beta;
}
model {
  alpha ~ normal(0, 100.); 
  theta ~ normal(0, 1); 
  beta ~ normal(0.0, 100.); 
  for (j in 1:N)  
    for (k in 1:T) 
      r[j, k] ~ bernoulli(inv_logit(beta * theta[j] - alpha[k]));
}

generated quantities {
  real mean_alpha; 
  real a[T]; 
  mean_alpha <- mean(alpha);
  for(t in 1:T) a[t] <- alpha[t] - mean_alpha;
} 

