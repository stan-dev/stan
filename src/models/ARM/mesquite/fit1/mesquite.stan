data {
  int<lower=0> N; 
  vector[N] Diam1;
  vector[N] Diam2;
  vector[N] CanHt; //canopy height
  vector[N] TotHt; //total height
  vector[N] Dens; //density
  vector[N] Group; //group (0 for MCD, 1 other)
  vector[N] LeafWt; //weight
}
parameters {
  vector[7] beta;
  real<lower=0> sigma;
} 
model {
  LeafWt ~ normal(beta[1] + beta[2] * Diam1 + beta[3] * Diam2 + beta[4] * CanHt 
                  + beta[5] * TotHt + beta[6] * Dens + beta[7] * Group, sigma);
}
