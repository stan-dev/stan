data {
  int<lower=0> N;
  vector[N] kid_score;
  vector[N] mom_hs;
}
transformed data {           // standardization
  vector[N] z_kid_score;
  vector[N] z_mom_hs;
  real kid_score_mean;
  real mom_hs_mean;
  real<lower=0> kid_score_sd;
  real<lower=0> mom_hs_sd;
 
  kid_score_mean <- mean(kid_score);
  mom_hs_mean    <- mean(mom_hs);

  kid_score_sd   <- sd(kid_score);
  mom_hs_sd      <- sd(mom_hs);

  z_kid_score    <- (kid_score - kid_score_mean) / kid_score_sd;
  z_mom_hs       <- (mom_hs - mom_hs_mean) / mom_hs_sd;
}
parameters {
  vector[2] z_beta;
  real<lower=0> z_sigma;
}
model {                      // vectorization
  z_kid_score ~ normal(z_beta[1] + z_beta[2] * z_mom_hs, z_sigma);
}
generated quantities {       // recovered parameter values
  vector[2] beta;
  real<lower=0> sigma;
  beta[1] <- kid_score_sd
             * (z_beta[1] - z_beta[2] * mom_hs_mean / mom_hs_sd)
             + kid_score_mean;
  beta[2] <- z_beta[2] * kid_score_sd / mom_hs_sd;
  sigma   <- kid_score_sd * z_sigma;
}
