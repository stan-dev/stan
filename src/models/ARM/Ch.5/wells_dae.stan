data {
  int<lower=0> N;
  vector[N] arsenic;
  vector[N] dist;
  vector[N] educ;
  int<lower=0,upper=1> switched[N];
}
transformed data {
  vector[N] dist100;         // rescaling
  vector[N] educ4;
  real arsenic_mean;         // centering
  real dist100_mean;         
  real educ4_mean;
  vector[N] c_arsenic;
  vector[N] c_dist100;
  vector[N] c_educ4;

  // rescaling
  dist100 <- dist / 100.0;
  educ4   <- educ / 4.0;
  // centering
  dist100_mean <- mean(dist100);
  arsenic_mean <- mean(arsenic);
  educ4_mean   <- mean(educ4);
  c_dist100    <- dist100 - dist100_mean;
  c_arsenic    <- arsenic - arsenic_mean;
  c_educ4      <- educ4 - educ4_mean;
}
parameters {
  vector[4] c_beta;
}
model {
  switched ~ bernoulli_logit(c_beta[1] + c_beta[2] * c_dist100 
                             + c_beta[3] * c_arsenic
                             + c_beta[4] * c_educ4);
}
generated quantities {       // recovered parameter values
  vector[4] beta;
  beta[4] <- c_beta[4];
  beta[3] <- c_beta[3];
  beta[2] <- c_beta[2];
  beta[1] <- c_beta[1] - beta[2] * dist100_mean - beta[3] * arsenic_mean
             - beta[4] * educ4_mean;
}
