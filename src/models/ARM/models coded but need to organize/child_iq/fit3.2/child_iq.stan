data {
  int<lower=0> N; 
  vector[N] kid_score;
  vector[N] mom_iq;
  vector[N] mom_hs;
}
 
transformed data {
  vector[N] c_mom_hs;
  vector[N] c_mom_iq;
  vector[N] inter;
  c_mom_hs <- mom_hs - 0.5;
  c_mom_iq <- mom_iq - 100;
  inter <- c_mom_hs .* c_mom_iq;
}

parameters {
  vector beta[4];
  real<lower=0> sigma;
} 

model {
  kid_score ~ normal(beta[1] + beta[2] * c_mom_hs + beta[3] * c_mom_iq
                          + beta[4] * inter, sigma);
}
