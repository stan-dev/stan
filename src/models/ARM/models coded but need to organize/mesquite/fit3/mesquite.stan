data {
  int<lower=0> N; 
  vector[N] Diam1;
  vector[N] Diam2;
  vector[N] CanHt; //canopy height
  vector[N] LeafWt; //weight
}
transformed data {
  vector[N] log_canopy_volume;
  vector[N] log_LeafWt; //log weight

  log_canopy_volume <- log(Diam1 .* Diam2 .* CanHt);
  log_LeafWt <- log(LeafWt);
}
parameters {
  vector[2] beta;
  real<lower=0> sigma;
} 
model {
  log_LeafWt ~ normal(beta[1] + beta[2] * log_canopy_volume, sigma);
}
