data {
  int<lower=0> N; 
  vector[N] kid_score;
  vector[N] mom_iq;
  vector[N] mom_hs;
}
 
transformed data {
  vector[N] inter;
  inter <- mom_hs .* mom_iq;
}

parameters {
  vector beta[4];
  real<lower=0> sigma;
} 

model {
  kid_score ~ normal(beta[1] + beta[2] * mom_hs + beta[3] * mom_iq
                          + beta[4] * inter, sigma);
}
