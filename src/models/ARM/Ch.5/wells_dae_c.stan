data {
  int<lower=0> N;
  int<lower=0,upper=1> switched[N];
  vector[N] dist;
  vector[N] arsenic;
  vector[N] educ;
}
transformed data {
  vector[N] c_dist100;
  vector[N] c_arsenic;
  vector[N] da_inter;
  vector[N] educ4;
  c_dist100 <- (dist - mean(dist)) / 100.0;
  c_arsenic <- arsenic - mean(arsenic);
  da_inter  <- c_dist100 .* c_arsenic;
  educ4     <- educ / 4.0;
}
parameters {
  vector[5] beta;
}
model {
  switched ~ bernoulli_logit(beta[1] + beta[2] * c_dist100 + beta[3] * c_arsenic
                              + beta[4] * da_inter + beta[5] * educ4);
}
