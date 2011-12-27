data {
  int(0,)  N;
  double y[N];
}
parameters {
  double(0,1) theta;
  double mu[2];
}
derived parameters {
  double(0,1) one_minus_theta;
}
model {
  theta ~ uniform(0,1);
  one_minus_theta <- 1.0 - theta;
  mu ~ normal(0,10);
  for (n in 1:N) {
    lp__ <- lp__ + log(theta * exp(normal_log(y[n],mu[1],1.0)
                       + (1.0 - theta) * exp(normal_log(y[n],mu[2],1.0))));
  }
}