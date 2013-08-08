data {
  int<lower=0> N;
  vector[N] canopy_height;
  vector[N] density;
  vector[N] diam1;
  vector[N] diam2;
  vector[N] group;
  vector[N] total_height;
  vector[N] weight;
}
transformed data {
  vector[N] log_canopy_height;
  vector[N] log_density;
  vector[N] log_diam1;
  vector[N] log_diam2;
  vector[N] log_total_height;
  vector[N] log_weight;

  log_canopy_height <- log(canopy_height);
  log_density       <- log(density);
  log_diam1         <- log(diam1);
  log_diam2         <- log(diam2);
  log_total_height  <- log(total_height);
  log_weight        <- log(weight);
}
parameters {
  vector[7] beta;
  real<lower=0> sigma;
}
model {
  log_weight ~ normal(beta[1] + beta[2] * log_diam1 + beta[3] * log_diam2
                  + beta[4] * log_canopy_height + beta[5] * log_total_height
                  + beta[6] * log_density + beta[7] * group,
                  sigma);
}
