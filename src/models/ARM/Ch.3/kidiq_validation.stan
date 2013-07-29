data {
  int<lower=0> N;
  vector[N] ppvt;
  vector[N] hs;
  vector[N] afqt;
}
transformed data {           // standardization
  vector[N] z_ppvt;
  vector[N] z_hs;
  vector[N] z_afqt;
  real ppvt_mean;
  real<lower=0> ppvt_sd;
  real hs_mean;
  real<lower=0> hs_sd;
  real afqt_mean;
  real<lower=0> afqt_sd;
  ppvt_mean <- mean(ppvt);
  ppvt_sd   <- sd(ppvt);
  hs_mean   <- mean(hs);
  hs_sd     <- sd(hs);
  afqt_mean <- mean(afqt);
  afqt_sd   <- sd(afqt);
  z_ppvt    <- (ppvt - ppvt_mean) / ppvt_sd;
  z_hs      <- (hs - hs_mean) / hs_sd;
  z_afqt    <- (afqt - afqt_mean) / afqt_sd;
}
parameters {
  vector[3] z_beta;
  real<lower=0> z_sigma;
}
model {                      // vectorization
  z_ppvt ~ normal(z_beta[1] + z_beta[2] * z_hs + z_beta[3] * z_afqt, z_sigma);
}
generated quantities {       // recovered parameter values
  vector[3] beta;
  real<lower=0> sigma;
  beta[1] <- ppvt_sd
             * (z_beta[1] - z_beta[2] * hs_mean / hs_sd
                - z_beta[3] * afqt_mean / afqt_sd)
             + ppvt_mean;
  beta[2] <- z_beta[2] * ppvt_sd / hs_sd;
  beta[3] <- z_beta[3] * ppvt_sd / afqt_sd;
  sigma   <- ppvt_sd * z_sigma;
}
