data {
  int<lower=0> N; 
  vector[N] dist;
  vector[N] arsenic;
  vector[N] educ;
  int<lower=0,upper=1> switch_w[N];
}
transformed data {
  vector[N] dist100;
  vector[N] c_dist100;
  vector[N] c_arsenic;
  vector[N] educ4;
  vector[N] c_educ4;
  vector[N] inter_dist_ars;
  vector[N] inter_dist_edu;
  vector[N] inter_ars_edu;
  real mu_dist100;
  real mu_arsenic;
  real mu_educ4;
  dist100 <- dist / 100;
  educ4 <- educ / 4;
  mu_dist100 <- mean(dist100);
  mu_arsenic <- mean(arsenic);
  mu_educ4 <- mean(educ4);
  c_dist100 <- dist100 - mu_dist100;
  c_arsenic <- arsenic - mu_arsenic;
  c_educ4 <- educ4 - mu_educ4;
  inter_dist_ars <- c_dist100 .* c_arsenic;
  inter_dist_edu <- c_dist100 .* c_educ4;
  inter_ars_edu <- c_arsenic .* c_educ4;
}
parameters {
  vector[7] beta;
} 
model {
  for (n in 1:N)
    switch_w[n] ~ bernoulli(inv_logit(beta[1] + beta[2] * c_dist100[n] 
                            + beta[3] * c_arsenic[n] + beta[4] * c_educ4[n]
                            + beta[5] * inter_dist_ars[n] 
                            + beta[6] * inter_dist_edu[n]
                            + beta[7] * inter_ars_edu[n]));
}
