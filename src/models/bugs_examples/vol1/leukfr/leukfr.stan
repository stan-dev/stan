data {
  int(0,) N;
  int(0,) NT;
  int(0,) obs_t[N]; 
  int(0,) t[NT + 1]; 
  int(0,) fail[N]; 
  int(0,) Npair; 
  int(0,) pair[Npair];
  double Z[N]; 
}

derived data {
  int(0,) Y[N, NT];
  int(0,) dN[N, NT]; 
  double c;
  double r; 
  for(i in 1:N) {
    for(j in 1:NT) {
      Y[i, j] <- step(obs_t[i] - t[j] + .000000001);
      dN[i, j] <- Y[i, j] * fail[i] * step(t[j + 1] - obs_t[i] - .000000001);
    }
  }
  c <- 0.001; 
  r <- 0.1; 
}

parameters {
  double beta; 
  double(0,) tau;
  double(0,) dL0[NT]; 
  double b[Npair]; 
} 

derived parameters {
  double(0,) sigma; 
  sigma <- 1 / sqrt(tau); 
} 

model {
  beta ~ normal(0, 1000);
  tau ~ gamma(.001, .001); 
  for (k in 1:Npair) b[k] ~ normal(0, sigma); 
  for(j in 1:NT) {
    dL0[j] ~ gamma(r * (t[j + 1] - t[j]) * c, c);
    for(i in 1:N) {
      dN[i, j] ~ poisson(Y[i, j] * exp(beta * Z[i] + b[pair[i]]) * dL0[j]); 
    }     
  }
}
