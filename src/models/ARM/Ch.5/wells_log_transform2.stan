data {
  int<lower=0> N; 
  vector[N] arsenic;
  vector[N] dist;
  vector[N] educ;
  int<lower=0,upper=1> switc[N];
}
transformed data {
  vector[N] dist100;
  vector[N] educ4;
  vector[N] log_arsenic;
  vector[N] inter_dist_ars;
  vector[N] inter_dist_edu;
  vector[N] inter_ars_edu;

  dist100 <- dist / 100.0;
  educ4 <- educ / 4.0;
  log_arsenic <- log(arsenic);

  inter_dist_ars <- dist100 .* log_arsenic;
  inter_dist_edu <- dist100 .* educ4;
  inter_ars_edu <- log_arsenic .* educ4;
}
parameters {
  vector[7] beta;
} 
model {
  switc ~ bernoulli_logit(beta[1] + beta[2] * dist100
                            + beta[3] * log_arsenic + beta[4] * educ4
                            + beta[5] * inter_dist_ars
                            + beta[6] * inter_dist_edu
                            + beta[7] * inter_ars_edu);
}
