data {
  int<lower=0> N;
  int<lower=0,upper=1> switched[N];
  vector[N] dist;
  vector[N] arsenic;
  vector[N] educ;
}
transformed data {
  vector[N] dist100;         // rescaling
  vector[N] educ4;
  real dist100_mean;         // centering
  real arsenic_mean;
  real educ4_mean;
  vector[N] c_arsenic;
  vector[N] c_dist100;
  vector[N] c_educ4;
  vector[N] c_inter;

  // rescaling
  dist100  <- dist / 100.0;
  educ4    <- educ / 4.0;
  // centering
  dist100_mean <- mean(dist100);
  arsenic_mean <- mean(arsenic);
  educ4_mean   <- mean(educ4);
  c_dist100    <- dist100 - dist100_mean;
  c_arsenic    <- arsenic - arsenic_mean;
  c_educ4      <- educ4 - educ4_mean;
  c_inter      <- c_dist100 .* c_arsenic;
}
parameters {
  vector[5] c_beta;
}
model {
  switched ~ bernoulli_logit(c_beta[1] + c_beta[2] * c_dist100 
                             + c_beta[3] * c_arsenic + c_beta[4] * c_educ4 
                             + c_beta[5] * c_inter);
}
generated quantities {       // recovered parameter values
  vector[5] beta;
  beta[5] <- c_beta[5];
  beta[4] <- c_beta[4];
  beta[3] <- c_beta[3] - beta[5] * dist100_mean;
  beta[2] <- c_beta[2] - beta[5] * arsenic_mean;
  beta[1] <- c_beta[1] - beta[2] * dist100_mean - beta[3] * arsenic_mean
             - beta[4] * educ4_mean - beta[5] * (dist100_mean * arsenic_mean);
}
