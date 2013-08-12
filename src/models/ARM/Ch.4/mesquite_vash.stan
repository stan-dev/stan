data {
  int<lower=0> N;
  vector[N] weight;
  vector[N] diam1;
  vector[N] diam2;
  vector[N] canopy_height;
  vector[N] total_height;
  vector[N] group;
}
transformed data {
  vector[N] log_weight;
  vector[N] log_canopy_volume;
  vector[N] log_canopy_area;
  vector[N] log_canopy_shape;
  vector[N] log_total_height;
  log_weight        <- log(weight);
  log_canopy_volume <- log(diam1 .* diam2 .* canopy_height);
  log_canopy_area   <- log(diam1 .* diam2);
  log_canopy_shape  <- log(diam1 ./ diam2);
  log_total_height  <- log(total_height);
}
parameters {
  vector[6] beta;
  real<lower=0> sigma;
}
model {
  log_weight ~ normal(beta[1] + beta[2] * log_canopy_volume
                      + beta[3] * log_canopy_area + beta[4] * log_canopy_shape
                      + beta[5] * log_total_height + beta[6] * group,
                      sigma);
}
