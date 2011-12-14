
data {
  int(0,) N; 
  int(0,) x[N];
  double  t[N]; 
} 

parameters {
  double(0,) alpha; 
  double(0,) beta;
  double(0,) theta[N];
} 

model {
  alpha ~ exponential(1.0);
  beta ~ gamma(0.1, 1.0);
  for (i in 1:N){
    theta[i] ~ gamma(alpha, beta);
    x[i] ~ poisson(theta[i] * t[i]);
  }
}
