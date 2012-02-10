transformed data {
  cov_matrix(2) S;

  for (i in 1:2)
    for (j in 1:2)
      S[i,j] <- 0.0;

  S[1,1] <- 2.0;
  S[2,2] <- 0.5;
}
parameters {
  double(-1,1) rho;
  double(0,) var1;
  double(0,) var2;
}
model {
  double cov;
  matrix(2,2) W;

  W[1,1] <- var1;
  W[2,2] <- var2;

  cov <- rho * sqrt(var1 * var2);
  lp__ <- lp__ + 0.5 * (log(var1) + log(var2));   

  W[1,2] <- cov;
  W[2,1] <- cov;
  W ~ wishart(5, S);
}
