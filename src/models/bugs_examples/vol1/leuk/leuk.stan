data {
  int(0,) N;
  int(0,) NT;
  int obs_t[N]; 
  int t[NT + 1]; 
  int fail[N]; 
  double Z[N]; 
}

derived data {
  int(0,) Y[N, NT];
  int(0,) dN[N, NT]; 
  double c;
  double r; 
  for(i in 1:N) {
    for(j in 1:NT) {
      Y[i, j] <- int_step(obs_t[i] - t[j] + .000000001);
      dN[i, j] <- Y[i, j] * fail[i] * int_step(t[j + 1] - obs_t[i] - .000000001);
    }
  }
  c <- 0.001; 
  r <- 0.1; 
}

parameters {
  double beta; 
  double(0,) dL0[NT]; 
} 

model {
  beta ~ normal(0, 10);
  for(j in 1:NT) {
    dL0[j] ~ gamma(r * (t[j + 1] - t[j]) * c, c);
    for(i in 1:N) {
       dN[i, j] ~ poisson(Y[i, j] * exp(beta * Z[i]) * dL0[j]); 
    }     
    // Survivor function = exp(-Integral{l0(u)du})^exp(beta*z)    
    // S.treat[j] <- pow(exp(-sum(dL0[1:j])), exp(beta * -0.5));
    // S.placebo[j] <- pow(exp(-sum(dL0[1:j])), exp(beta * 0.5));	
  }
}
