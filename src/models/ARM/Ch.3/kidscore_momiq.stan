data {
  int<lower=0> N;
  vector[N] kid_score;
  vector[N] mom_iq;
}
transformed data {           // standardization
  vector[N] z_kid_score;
  vector[N] z_mom_iq;
  real kid_score_mean;
  real<lower=0> kid_score_sd;
  real mom_iq_mean;
  real<lower=0> mom_iq_sd;
  kid_score_mean <- mean(kid_score);
  kid_score_sd   <- sd(kid_score);
  mom_iq_mean    <- mean(mom_iq);
  mom_iq_sd      <- sd(mom_iq);
  z_kid_score    <- (kid_score - kid_score_mean) / kid_score_sd;
  z_mom_iq       <- (mom_iq - mom_iq_mean) / mom_iq_sd;
}
parameters {
  vector[2] z_beta;
  real<lower=0> z_sigma;
}
model {                      // vectorization
  z_kid_score ~ normal(z_beta[1] + z_beta[2] * z_mom_iq, z_sigma);
}
generated quantities {       // recovered parameter values
  vector[2] beta;
  real<lower=0> sigma;
  beta[1] <- kid_score_sd
             * (z_beta[1] - z_beta[2] * mom_iq_mean / mom_iq_sd)
             + kid_score_mean;
  beta[2] <- z_beta[2] * kid_score_sd / mom_iq_sd;
  sigma   <- kid_score_sd * z_sigma;
}
