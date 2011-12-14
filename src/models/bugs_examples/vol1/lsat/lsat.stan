data {
  int(0,) N;
  int(0,) R;
  int(0,) T;
  int(0,) culm[R];
  int(0,) response[R,T];
}

derived data {
  int(0,) r[N,T];
  
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
  double alpha[T];
  double theta[N];
  double(0,) beta;
}

//derived parameters {
//  double mean_alpha;
//  double a[T];
//}

model {
  // prior
  for (k in 1:T) {
    alpha[k] ~ normal(0.0,1.0E4);
  }

//  for(k in 1:T) {
//    mean_alpha <- mean(alpha);
//    a[k] <- alpha[k] - mean_alpha;
//   }
  
  for(j in 1:N) { 
    theta[j] ~ normal(0,1);
  }
  
  beta ~ normal(0.0,1.0E4);
  // Rasch model
  for (j in 1:N) {
    for (k in 1:T) {
      r[j,k] ~ bernoulli(inv_logit(beta*theta[j] - alpha[k]));
    }
  }
}

