# Equiv: bioequivalence in a cross-over trial
#  http://www.openbugs.info/Examples/Equiv.html

data {
  int(0,) P; 
  int(0,) N; 
  int group[N]; 
  double Y[N, P]; 
  int sign[2];

  int T[N, P]; 
} 

// If T is not in the data, using the following transformed
// data:
// 
#   transformed data {
#     int T[N, P]; 
#     for (n in 1:N) for (p in 1:P)  
#       // (group[n] * (p - 1.5) + 1.5)  is 1/2, but of type double, 
#       // using int_step as a workaround 
#       T[n, p] <- int_step(group[n] * (p - 1.5) + .4) + 1; 
#   } 

parameters {
  double mu;
  double phi; 
  double pi; 
  double(0,) sigmasq; 
  double(0,) sigmasq2;
  double delta[N]; 
} 

transformed parameters {
  double sigma; 
  sigma <- sqrt(sigmasq); 
} 

model {
  for (p in 1:P) {
    for (n in 1:N) {
      Y[n, p] ~ normal(mu + delta[n] + sign[T[n, p]] * phi / 2 + sign[p] * pi / 2., sigma); 
    }
  }
  delta ~ normal(0, sqrt(sigmasq2)); 
  sigmasq ~ inv_gamma(.001, .001); 
  sigmasq2 ~ inv_gamma(.001, .001); 
  mu ~ normal(0.0, 1000); 
  phi ~ normal(0.0, 1000); 
  pi ~ normal(0.0, 1000); 
  // theta <- exp(phi)
  // equiv <- step(theta - 0.8) - step(theta - 1.2)
}
