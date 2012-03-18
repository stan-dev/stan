data { 
  int(0,) N; 
  int(0,1) y[N];
} 
parameters {
  real(0,1) theta;
} 
model {
  for (n in 1:N) 
    y[n] ~ bernoulli(theta);
}
