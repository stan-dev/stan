data {
  int<lower=0> N;
  vector[N] earn;
  vector[N] height;
}
transformed data {
  vector[N] log10_earn;      
  vector[N] z_log10_earn;    
  vector[N] z_height;
  real log10_earn_mean;
  real height_mean;
  real<lower=0> log10_earn_sd;
  real<lower=0> height_sd;

  for (i in 1:N) {                       // log 10 transformation
    log10_earn[i] <- log10(earn[i]);
  }

  log10_earn_mean <- mean(log10_earn);   // standardization
  height_mean     <- mean(height);

  log10_earn_sd   <- sd(log10_earn);
  height_sd       <- sd(height);

  z_log10_earn    <- (log10_earn - log10_earn_mean) / log10_earn_sd;
  z_height        <- (height - height_mean) / height_sd;
}
parameters {
  vector[2] z_beta;
  real<lower=0> z_sigma;
}
model {                      // vectorization
  z_log10_earn ~ normal(z_beta[1] + z_beta[2] * z_height, z_sigma);
}
generated quantities {       // recovered parameter values
  vector[2] beta;
  real<lower=0> sigma;
  beta[1] <- log10_earn_sd
             * (z_beta[1] - z_beta[2] * height_mean / height_sd)
             + log10_earn_mean;
  beta[2] <- z_beta[2] * log10_earn_sd / height_sd;
  sigma   <- log10_earn_sd * z_sigma;
}
