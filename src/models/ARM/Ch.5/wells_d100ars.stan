data {
  int<lower=0> N;
  vector[N] arsenic;
  vector[N] dist;
  int<lower=0,upper=1> switched[N];
}
transformed data {
  vector[N] dist100;         // rescaling
  real dist100_mean;         // centering
  real arsenic_mean;
  vector[N] c_arsenic;
  vector[N] c_dist100;

  // rescaling
  dist100 <- dist / 100.0;
  // centering
  dist100_mean <- mean(dist100);
  arsenic_mean <- mean(arsenic);
  c_dist100 <- dist100 - dist100_mean;
  c_arsenic <- arsenic - arsenic_mean;
}
parameters {
  vector[3] c_beta;
}
model {
  switched ~ bernoulli_logit(c_beta[1] + c_beta[2] * c_dist100 
                             + c_beta[3] * c_arsenic);
}
generated quantities {       // recovered parameter values
  vector[3] beta;
  beta[3] <- c_beta[3];
  beta[2] <- c_beta[2];
  beta[1] <- c_beta[1] - beta[2] * dist100_mean - beta[3] * arsenic_mean;
}
