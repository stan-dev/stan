data {
  int(0,) N;
  int(0,) T;
  int obs_t[N]; 
  int t[T]; 
  double(0,) eps; 
  int fail[N]; 
  double Z[N]; 
}

derived data {
  double Y[N, T];
  double dN[N, T]; 
  double c;
  double r; 
  for(i in 1:N) {
    for(j in 1:T) {
      // risk set = 1 if obs_t >= t
      Y[i,j] <- step(obs_t[i] - t[j] + eps);
      dN[i, j] <- Y[i, j] * fail[i] * step(t[j + 1] - obs_t[i] - eps);
    }
  }
  c <- 0.001; 
  r <- 0.1; 
}

parameters {
  double beta; 
  double dL0[T]; 
} 

model {
  beta ~ normal(0, 1000);
  for(j in 1:T) {
    dL0[j] ~ gamma(r * (t[j + 1]- t[j]) * c, c);
    for(i in 1:N) {
       dN[i,j]   ~ poisson(Y[i,j] * exp(beta * Z[i]) * dL0[j]); 
    }     
    // Survivor function = exp(-Integral{l0(u)du})^exp(beta*z)    
    // S.treat[j] <- pow(exp(-sum(dL0[1:j])), exp(beta * -0.5));
    // S.placebo[j] <- pow(exp(-sum(dL0[1:j])), exp(beta * 0.5));	
  }
}
