data {
  int<lower=0> N; 
  vector[N] dist;
  vector[N] arsenic;
  vector[N] educ;
  int<lower=0,upper=1> switch_w[N];
}
transformed data {
  vector[N] dist100;
  vector[N] educ4;
  vector[N] inter_dist_ars;
  dist100 <- dist / 100;
  educ4 <- educ / 4;
  inter_dist_ars <- dist100 .* arsenic;
}
parameters {
  vector[5] beta;
} 
model {
  for (n in 1:N)
    switch_w[n] ~ bernoulli(inv_logit(beta[1] + beta[2] * dist100[n] 
                            + beta[3] * arsenic[n] + beta[4] * educ4[n]
                            + beta[5] * inter_dist_ars[n]));
}
