data {
  int<lower=0> N;
  vector[N] arsenic;
  vector[N] dist;
  int<lower=0,upper=1> switched[N];
}
transformed data {
  real arsenic_mean;  
  vector[N] dist100;         // rescaling
  real dist100_mean;         // centering
  vector[N] c_arsenic;
  vector[N] c_dist100;  
  vector[N] c_inter;

  // rescaling
  dist100 <- dist / 100.0;
  // centering
  dist100_mean <- mean(dist100);
  arsenic_mean <- mean(arsenic);
  c_dist100 <- dist100 - dist100_mean;
  c_arsenic <- arsenic - arsenic_mean;
  c_inter <- c_dist100 .* c_arsenic;
}
parameters {
  vector[4] c_beta;
}
model {
  switched ~ bernoulli_logit(c_beta[1] + c_beta[2] * c_dist100 
                             + c_beta[3] * c_arsenic + c_beta[4] * c_inter);
}
generated quantities {       // recovered parameter values
  vector[4] beta;
  beta[4] <- c_beta[4];
  beta[3] <- c_beta[3] - beta[4] * dist100_mean;
  beta[2] <- c_beta[2] - beta[4] * arsenic_mean;
  beta[1] <- c_beta[1] - beta[2] * dist100_mean - beta[3] * arsenic_mean
             - beta[4] * (dist100_mean * arsenic_mean);
}
