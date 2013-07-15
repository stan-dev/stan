data {
  int<lower=0> N; 
  vector[N] Diam1;
  vector[N] Diam2;
  vector[N] CanHt; //canopy height
  vector[N] TotHt; //total height
  vector[N] Group; //group (0 for MCD, 1 other)
  vector[N] LeafWt; //weight
}
transformed data {
  vector[N] log_CanArea;
  vector[N] log_CanShape;
  vector[N] log_CanVol;
  vector[N] log_TotHt; //log total height
  vector[N] log_LeafWt; //log weight

  log_CanArea <- log(Diam1 .* Diam2);
  log_CanShape <- log(Diam1 ./ Diam2);
  log_CanVol <- log(Diam1 .* Diam2 .* CanHt);
  log_TotHt <- log(TotHt);
  log_LeafWt <- log(LeafWt);
}
parameters {
  vector[6] beta;
  real<lower=0> sigma;
} 
model {
  log_LeafWt ~ normal(beta[1] + beta[2] * log_CanVol + beta[3] * log_CanArea
                      + beta[4] * log_CanShape + beta[5] * log_TotHt
                      + beta[6] * Group, sigma);
}
