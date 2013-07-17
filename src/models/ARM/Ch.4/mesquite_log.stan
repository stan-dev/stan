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
transformed data {
  vector[N] log_Diam1;
  vector[N] log_Diam2;
  vector[N] log_CanHt; //log canopy height
  vector[N] log_TotHt; //log total height
  vector[N] log_Dens; //log density
  vector[N] log_LeafWt; //log weight

  log_Diam1 <- log(Diam1);
  log_Diam2 <- log(Diam2);
  log_CanHt <- log(CanHt);
  log_TotHt <- log(TotHt);
  log_Dens <- log(Dens);
  log_LeafWt <- log(LeafWt);
}
parameters {
  vector[7] beta;
  real<lower=0> sigma;
} 
model {
  log_LeafWt ~ normal(beta[1] + beta[2] * log_Diam1 + beta[3] * log_Diam2 + beta[4] * log_CanHt 
                  + beta[5] * log_TotHt + beta[6] * log_Dens + beta[7] * Group, sigma);
}
