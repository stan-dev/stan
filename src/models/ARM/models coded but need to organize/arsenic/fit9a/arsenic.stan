data {
  int<lower=0> N; 
  vector[N] dist;
  vector[N] arsenic;
  vector[N] educ;
  int<lower=0,upper=1> switch_w[N];
}
transformed data {
  vector[N] dist100;
  vector[N] log_arsenic;
  vector[N] educ4;
  vector[N] inter_dist_ars;
  vector[N] inter_dist_edu;
  vector[N] inter_ars_edu;
  dist100 <- dist / 100;
  educ4 <- educ / 4;
  log_arsenic <- log(arsenic);
  inter_dist_ars <- dist100 .* log_arsenic;
  inter_dist_edu <- dist100 .* educ4;
  inter_ars_edu <- log_arsenic .* educ4;
}
parameters {
  vector[7] beta;
} 
model {
  for (n in 1:N)
    switch_w[n] ~ bernoulli(inv_logit(beta[1] + beta[2] * dist100[n] 
                            + beta[3] * log_arsenic[n] + beta[4] * educ4[n]
                            + beta[5] * inter_dist_ars[n] 
                            + beta[6] * inter_dist_edu[n]
                            + beta[7] * inter_ars_edu[n]));
}
