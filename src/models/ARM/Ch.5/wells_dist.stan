data {
  int<lower=0> N;
  vector[N] dist;
  int<lower=0,upper=1> switched[N];
}
transformed data {           // centering
  real dist_mean;
  vector[N] c_dist;

  dist_mean <- mean(dist);
  c_dist    <- dist - dist_mean;
}
parameters {
  vector[2] c_beta;
}
model {
  switched ~ bernoulli_logit(c_beta[1] + c_beta[2] * c_dist);
}
generated quantities {       // recovered parameter values
  vector[2] beta;
  beta[2] <- c_beta[2];
  beta[1] <- c_beta[1] - beta[2] * dist_mean;
}
