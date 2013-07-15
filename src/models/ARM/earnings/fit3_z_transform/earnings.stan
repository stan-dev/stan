data {
  int<lower=0> N; 
  vector[N] earnings;
  vector[N] height;
  vector[N] sex;
} 
transformed data {
  vector[N] log_earnings;
  vector[N] height_male_inter;
  vector[N] male;
  vector[N] z_height;
  real mu;
  real sig;
  mu <- mean(height);
  sig <- 2.0 * sd(height);
  z_height <- (height - mu) / sig;
  log_earnings <- log(earnings);
  male <- 2 - sex;
  height_male_inter <- z_height .* male;
}
parameters {
  vector[4] beta;
  real<lower=0> sigma;
}
model {
  log_earnings ~ normal(beta[1] + beta[2] * z_height + beta[3] * male 
                             + beta[4] * height_male_inter, sigma);
}
