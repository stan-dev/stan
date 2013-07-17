data {
  int<lower=0> N; 
  vector[N] dist;
  vector[N] arsenic;
  vector[N] educ;
  int<lower=0,upper=1> switc[N];
}
transformed data {
  vector[N] dist100;
  vector[N] c_dist100;
  vector[N] log_arsenic;
  vector[N] c_log_arsenic;
  vector[N] educ4;
  vector[N] c_educ4;
  vector[N] inter_dist_ars;
  vector[N] inter_dist_edu;
  vector[N] inter_ars_edu;
  real mu_dist100;
  real mu_log_arsenic;
  real mu_educ4;
  dist100 <- dist / 100.0;
  educ4 <- educ / 4.0;
  log_arsenic <- log(arsenic);
  mu_dist100 <- mean(dist100);
  mu_log_arsenic <- mean(log_arsenic);
  mu_educ4 <- mean(educ4);
  c_dist100 <- dist100 - mu_dist100;
  c_log_arsenic <- log_arsenic - mu_log_arsenic;
  c_educ4 <- educ4 - mu_educ4;
  inter_dist_ars <- c_dist100 .* c_log_arsenic;
  inter_dist_edu <- c_dist100 .* c_educ4;
  inter_ars_edu <- c_log_arsenic .* c_educ4;
}
parameters {
  vector[7] beta;
} 
model {
  switc ~ bernoulli_logit(beta[1] + beta[2] * c_dist100
                            + beta[3] * c_log_arsenic + beta[4] * c_educ4
                            + beta[5] * inter_dist_ars 
                            + beta[6] * inter_dist_edu
                            + beta[7] * inter_ars_edu);
}
