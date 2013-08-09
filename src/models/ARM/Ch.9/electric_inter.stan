data {
  int<lower=0> N;
  vector[N] post_test;
  vector[N] pre_test;
  vector[N] treatment;
}
transformed data {
  vector[N] inter;           // interaction
  real inter_mean;
  real post_test_mean;       // standardization
  real pre_test_mean;
  real treatment_mean;
  real<lower=0> inter_sd;
  real<lower=0> post_test_sd;
  real<lower=0> pre_test_sd;
  real<lower=0> treatment_sd;
  vector[N] z_inter;
  vector[N] z_post_test;
  vector[N] z_pre_test;
  vector[N] z_treatment;

  // interactin
  inter          <- treatment .* pre_test;
  // standardization
  inter_mean     <- mean(inter);
  pre_test_mean  <- mean(pre_test);
  post_test_mean <- mean(post_test);
  treatment_mean <- mean(treatment);
  inter_sd       <- sd(inter);
  pre_test_sd    <- sd(pre_test);
  post_test_sd   <- sd(post_test);
  treatment_sd   <- sd(treatment);
  z_inter        <- (inter - inter_mean) / inter_sd;
  z_pre_test     <- (pre_test - pre_test_mean) / pre_test_sd;
  z_post_test    <- (post_test - post_test_mean) / post_test_sd;
  z_treatment    <- (treatment - treatment_mean) / treatment_sd;
}
parameters {
  vector[4] z_beta;
  real<lower=0> z_sigma;
}
model {
  z_post_test ~ normal(z_beta[1] + z_beta[2] * z_treatment 
                       + z_beta[3] * z_pre_test
                       + z_beta[4] * z_inter, z_sigma);
}
generated quantities {       // recovered parameter values
  vector[4] beta;
  real<lower=0> sigma;
  beta[1] <- post_test_sd
             * (z_beta[1] - z_beta[2] * treatment_mean / treatment_sd
                - z_beta[3] * pre_test_mean / pre_test_sd
                - z_beta[4] * inter_mean / inter_sd)
             + post_test_mean;
  beta[2] <- z_beta[2] * post_test_sd / treatment_sd;
  beta[3] <- z_beta[3] * post_test_sd / pre_test_sd;
  beta[4] <- z_beta[4] * post_test_sd / inter_sd;
  sigma   <- post_test_sd * z_sigma;
}
