data {
  int<lower=0> N;
  vector[N] kid_score;
  vector[N] mom_hs;
  vector[N] mom_iq;
  real mom_hs_new;           // for prediction
  real mom_iq_new;
}
transformed data {           // standardization
  vector[N] z_kid_score;
  vector[N] z_mom_hs;
  vector[N] z_mom_iq;
  real kid_score_mean;
  real mom_hs_mean;
  real mom_iq_mean;
  real<lower=0> kid_score_sd;
  real<lower=0> mom_hs_sd;
  real<lower=0> mom_iq_sd;

  kid_score_mean <- mean(kid_score);
  mom_hs_mean    <- mean(mom_hs);
  mom_iq_mean    <- mean(mom_iq);

  mom_hs_sd      <- sd(mom_hs);
  kid_score_sd   <- sd(kid_score);
  mom_iq_sd      <- sd(mom_iq);

  z_kid_score    <- (kid_score - kid_score_mean) / kid_score_sd;
  z_mom_hs       <- (mom_hs - mom_hs_mean) / mom_hs_sd;
  z_mom_iq       <- (mom_iq - mom_iq_mean) / mom_iq_sd;
}
parameters {
  vector[3] z_beta;
  real<lower=0> z_sigma;
}
model {                      // vectorization
  z_kid_score ~ normal(z_beta[1] + z_beta[2] * z_mom_hs + z_beta[3] * z_mom_iq,
                       z_sigma);
}
generated quantities {
  vector[3] beta;
  real<lower=0> sigma;
  real kid_score_pred;
  // recovered parameter values
  beta[1] <- kid_score_sd
             * (z_beta[1] - z_beta[2] * mom_hs_mean / mom_hs_sd
                - z_beta[3] * mom_iq_mean / mom_iq_sd)
             + kid_score_mean;
  beta[2] <- z_beta[2] * kid_score_sd / mom_hs_sd;
  beta[3] <- z_beta[3] * kid_score_sd / mom_iq_sd;
  sigma   <- kid_score_sd * z_sigma;
  // prediction
  kid_score_pred <- normal_rng(beta[1] + beta[2] * mom_hs_new 
                               + beta[3] * mom_iq_new, sigma);
}
