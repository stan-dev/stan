data {
  int<lower=0> J;
  int<lower=0> y[J];
  int<lower=0> n[J];
}
parameters {
  real<lower=0,upper=1> theta[J];
  real<lower=0,upper=1> lambda;
  real<lower=0> kappa;

}
transformed parameters {
  real<lower=0> alpha;
  real<lower=0> beta;
  alpha <- lambda * kappa;
  beta <- (1 - lambda) * kappa;
}
model {
  kappa ~ pareto(0.1,2.5);
  theta ~ beta(alpha,beta);
  y ~ binomial(n,theta);
}
// generated quantities {
//   real avg;
//   int<lower=0,upper=1> above_avg[J];
//   avg <- mean(theta);
//   for (j in 1:J)
//     above_avg[j] <- (theta[j] > avg);
// }
