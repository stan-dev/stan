data {
  int<lower=0> N;
  vector[N] kid_score;
  vector[N] mom_hs;
  vector[N] mom_iq;
}
transformed data {           // standardization (SML, §18.5)
  vector[N] inter;
  vector[N] z_kid_score;
  vector[N] z_mom_hs;
  vector[N] z_mom_iq;
  vector[N] z_inter;
  real inter_mean;
  real kid_score_mean;
  real mom_hs_mean;
  real mom_iq_mean;
  real<lower=0> inter_sd;
  real<lower=0> kid_score_sd;
  real<lower=0> mom_hs_sd;
  real<lower=0> mom_iq_sd;

  inter          <- mom_hs .* mom_iq;

  inter_mean     <- mean(inter);
  kid_score_mean <- mean(kid_score);
  mom_hs_mean    <- mean(mom_hs);
  mom_iq_mean    <- mean(mom_iq);

  inter_sd       <- sd(inter);
  kid_score_sd   <- sd(kid_score);
  mom_hs_sd      <- sd(mom_hs);
  mom_iq_sd      <- sd(mom_iq);

  z_inter        <- (inter - inter_mean) / inter_sd;
  z_kid_score    <- (kid_score - kid_score_mean) / kid_score_sd;
  z_mom_hs       <- (mom_hs - mom_hs_mean) / mom_hs_sd;
  z_mom_iq       <- (mom_iq - mom_iq_mean) / mom_iq_sd;
}
parameters {
  vector[4] z_beta;
  real<lower=0> z_sigma;
}
model {                      // vectorization (SML, §18.2)
  z_kid_score ~ normal(z_beta[1] + z_beta[2] * z_mom_hs 
                       + z_beta[3] * z_mom_iq + z_beta[4] * z_inter,
                       z_sigma);
}
generated quantities {       // recovered parameter values (SML, §18.5)
  vector[4] beta;
  real<lower=0> sigma;
  beta[1] <- kid_score_sd
             * (z_beta[1] - z_beta[2] * mom_hs_mean / mom_hs_sd
                - z_beta[3] * mom_iq_mean / mom_iq_sd
                - z_beta[4] * inter_mean / inter_sd)
             + kid_score_mean;
  beta[2] <- z_beta[2] * kid_score_sd / mom_hs_sd;
  beta[3] <- z_beta[3] * kid_score_sd / mom_iq_sd;
  beta[4] <- z_beta[4] * kid_score_sd / inter_sd;
  sigma   <- kid_score_sd * z_sigma;
}
