data {
  int<lower=0> N; 
  vector[N] kid_score;
  vector[N] mom_iq;
  vector[N] mom_hs;
}
 
transformed data {
  vector[N] c_mom_hs;
  vector[N] c_mom_iq;
  real mean_mom_hs;
  real sd_mom_hs;
  real mean_mom_iq;
  real sd_mom_iq;
  mean_mom_hs <- mean(mom_hs);
  sd_mom_hs <- 2.0 * sd(mom_hs);
  mean_mom_iq <- mean(mom_iq);
  sd_mom_iq <- 2.0 * sd(mom_iq);
  z_mom_hs <- (mom_hs - mean_mom_hs) / sd_mom_hs;
  z_mom_iq <- (mom_iq - mean_mom_iq) / sd_mom_iq;
}

parameters {
  vector beta[3];
  real<lower=0> sigma;
} 

model {
  kid_score ~ normal(beta[1] + beta[2] * z_mom_hs + beta[3] * z_mom_iq, sigma);
}
