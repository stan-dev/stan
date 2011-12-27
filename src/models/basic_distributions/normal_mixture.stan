derived data {
  double(0,1) theta;
  double mu[2];
  double(0,) sigma[2];
  vector(3) v1;
  row_vector(3) v1_t;
  
  mu[1] <- 0.0;    sigma[1] <- 0.5;
  mu[2] <- 4.0;    sigma[2] <- 3.0;
  theta <- 0.25;
}
parameters {
  double y;
}
model {
  lp__ <- lp__ + log_sum_exp(log(theta) + normal_log(y,mu[1],sigma[1]),
                             log(1.0 - theta) + normal_log(y,mu[2],sigma[2]));
  v1_t <- transpose(v1);
}