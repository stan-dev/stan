data {
  int<lower=0> N;
  vector[N] earn;
  vector[N] height;
}
transformed data {           // standardization
  vector[N] z_earn;
  vector[N] z_height;
  real earn_mean;
  real<lower=0> earn_sd;
  real height_mean;
  real<lower=0> height_sd;
  earn_mean   <- mean(earn);
  earn_sd     <- sd(earn);
  height_mean <- mean(height);
  height_sd   <- sd(height);
  z_earn      <- (earn - earn_mean) / earn_sd;
  z_height    <- (height - height_mean) / height_sd;
}
parameters {
  vector[2] z_beta;
  real<lower=0> z_sigma;
}
model {                      // vectorization
  z_earn ~ normal(z_beta[1] + z_beta[2] * z_height, z_sigma);
}
generated quantities {       // recovered parameter values
  vector[2] beta;
  real<lower=0> sigma;
  beta[1] <- earn_sd
             * (z_beta[1] - z_beta[2] * height_mean / height_sd)
             + earn_mean;
  beta[2] <- z_beta[2] * earn_sd / height_sd;
  sigma   <- earn_sd * z_sigma;
}
