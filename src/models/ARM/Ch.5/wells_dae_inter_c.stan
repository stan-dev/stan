data {
  int<lower=0> N;
  vector[N] arsenic;
  vector[N] dist;
  vector[N] educ;
  int<lower=0,upper=1> switched[N];
}
transformed data {
  vector[N] c_arsenic;
  vector[N] c_dist100;
  vector[N] c_educ4;
  vector[N] ae_inter;
  vector[N] da_inter;
  vector[N] de_inter;

  c_dist100 <- (dist - mean(dist)) / 100.0;
  c_arsenic <- arsenic - mean(arsenic);
  c_educ4   <- (educ - mean(educ)) / 4.0;
  da_inter  <- c_dist100 .* c_arsenic;
  de_inter  <- c_dist100 .* c_educ4;
  ae_inter  <- c_arsenic .* c_educ4;
}
parameters {
  vector[7] beta;
}
model {
  switched ~ bernoulli_logit(beta[1] + beta[2] * c_dist100 + beta[3] * c_arsenic
                              + beta[4] * c_educ4 + beta[5] * da_inter
                              + beta[6] * de_inter + beta[7] * ae_inter);
}
