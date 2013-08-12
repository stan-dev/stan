data {
  int<lower=0> N;
  vector[N] kid_score;
  vector[N] mom_hs;
  vector[N] mom_iq;
  real mom_hs_new;           // for prediction
  real mom_iq_new;
}
parameters {
  vector[3] beta;
  real<lower=0> sigma;
}
model {
  kid_score ~ normal(beta[1] + beta[2] * mom_hs + beta[3] * mom_iq, sigma);
}
generated quantities {       // prediction
  real kid_score_pred;
  kid_score_pred <- normal_rng(beta[1] + beta[2] * mom_hs_new 
                               + beta[3] * mom_iq_new, sigma);
}
