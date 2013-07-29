data {
  int<lower=0> N;
  vector[N] earn;
  vector[N] height;
  vector[N] male;
}
transformed data {
  vector[N] log_earn;
  vector[N] log_height;
  vector[N] z_log_earn;      
  vector[N] z_log_height;
  vector[N] z_male;
  real log_earn_mean;
  real<lower=0> log_earn_sd;
  real log_height_mean;
  real<lower=0> log_height_sd;
  real male_mean;
  real<lower=0> male_sd;
  log_earn        <- log(earn);        // log transformation
  log_height      <- log(height);
  log_earn_mean   <- mean(log_earn);   // standardization
  log_earn_sd     <- sd(log_earn);
  log_height_mean <- mean(log_height);
  log_height_sd   <- sd(log_height);
  male_mean       <- mean(male);
  male_sd         <- sd(male);
  z_log_earn      <- (log_earn - log_earn_mean) / log_earn_sd;
  z_log_height    <- (log_height - log_height_mean) / log_height_sd;
  z_male          <- (male - male_mean) / male_sd;
}
parameters {
  vector[3] z_beta;
  real<lower=0> z_sigma;
}
model {                      // vectorization
  z_log_earn ~ normal(z_beta[1] + z_beta[2] * z_log_height + z_beta[3] * z_male,
                      z_sigma);
}
generated quantities {       // recovered parameter values
  vector[3] beta;
  real<lower=0> sigma;
  beta[1] <- log_earn_sd
             * (z_beta[1] - z_beta[2] * log_height_mean / log_height_sd
                - z_beta[3] * male_mean / male_sd)
             + log_earn_mean;
  beta[2] <- z_beta[2] * log_earn_sd / log_height_sd;
  beta[3] <- z_beta[3] * log_earn_sd / male_sd;
  sigma   <- log_earn_sd * z_sigma;
}
