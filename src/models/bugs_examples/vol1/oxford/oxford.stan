# http://www.mrc-bsu.cam.ac.uk/bugs/winbugs/Vol1.pdf
# Page 34: Oxford: smooth fit to log-odds ratios

data {
  int(0,) K; 
  int(0,) n0[K];
  int(0,) n1[K]; 
  int(0,) r0[K]; 
  int(0,) r1[K]; 
  int year[K]; 
} 
derived data {
  int yearsq[K]; 
  for (i in 1:K) 
    yearsq[i] <- year[i] * year[i]; 
} 
parameters {
  double mu[K]; 
  double alpha;
  double beta1; 
  double beta2;
  double(0, 3) sigma; // Q: do we need 'double(0, 3)' or just 'double'? 
  double b[K]; 
}
model {
  for (i in 1:K) {
    r0[i] ~ binomial(n0[i], inv_logit(mu[i])); 
    r1[i] ~ binomial(n1[i], 
                     inv_logit(mu[i] + alpha + beta1 * year[i] + beta2 * (yearsq[i] - 22) + sigma * b[i]));
    b[i]  ~ normal(0, 1);
    mu[i] ~ normal(0, 1000); 
  }

  alpha  ~ normal(0.0, 1000); 
  beta1  ~ normal(0.0, 1000); 
  beta2  ~ normal(0.0, 1000); 
  sigma  ~ uniform(0, 3);
}
