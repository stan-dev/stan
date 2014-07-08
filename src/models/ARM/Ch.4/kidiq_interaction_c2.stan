data {
  int<lower=0> N;
  vector[N] kid_score;
  vector[N] mom_hs;
  vector[N] mom_iq;
}
transformed data {           // centering on reference points
  vector[N] c2_mom_hs;
  vector[N] c2_mom_iq;
  vector[N] inter;
  c2_mom_hs <- mom_hs - 0.5;
  c2_mom_iq <- mom_iq - 100;  
  inter     <- c2_mom_hs .* c2_mom_iq;
}
parameters {
  vector[4] beta;
  real<lower=0> sigma;
}
model {
  kid_score ~ normal(beta[1] + beta[2] * c2_mom_hs + beta[3] * c2_mom_iq 
                     + beta[4] * inter, sigma);
}
