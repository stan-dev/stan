data {
  int<lower=0> N;
  vector[N] dist;
  int<lower=0,upper=1> switched[N];
}
transformed data {
  vector[N] dist100;         // rescaling
  real dist100_mean;         // centering
  vector[N] c_dist100;

  // rescaling
  dist100 <- dist / 100.0;   
  // centering
  dist100_mean <- mean(dist100);
  c_dist100    <- dist100 - dist100_mean;
}
parameters {
  vector[2] c_beta;
}
model {
  switched ~ bernoulli_logit(c_beta[1] + c_beta[2] * c_dist100);
}
generated quantities {       // recovered parameter values
  vector[2] beta;
  beta[2] <- c_beta[2];
  beta[1] <- c_beta[1] - beta[2] * dist100_mean;
}
