data {
  int<lower=0> N;
  vector[N] income;
  int<lower=0,upper=1> vote[N];
}
transformed data {           // centering
  real income_mean;
  vector[N] c_income;

  income_mean <- mean(income);
  c_income    <- income - income_mean;
}
parameters {
  vector[2] c_beta;
}
model {
  vote ~ bernoulli_logit(c_beta[1] + c_beta[2] * c_income);
}
generated quantities {       // recovered parameter values
  vector[2] beta;
  beta[2] <- c_beta[2];
  beta[1] <- c_beta[1] - beta[2] * income_mean;
}
