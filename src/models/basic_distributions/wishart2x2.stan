// Sample from 2 x 2 Wishart
// calculate matrix directly through non-matrix parameters

// WARNING:  
// This simple parameterization only works for 2 x 2
// matrices because positive definiteness is simple.  

transformed data {
  cov_matrix(2) S;

  for (i in 1:2)
    for (j in 1:2)
      S[i,j] <- 0.0;

  S[1,1] <- 2.0;
  S[2,2] <- 0.5;
}
parameters {
  real(-1,1) rho;
  real(0,) var1;
  real(0,) var2;
}
model {
  real cov;
  matrix(2,2) W;

  cov <- rho * sqrt(var1 * var2);

  W[1,1] <- var1;
  W[2,2] <- var2;
  W[1,2] <- cov;
  W[2,1] <- cov;

  // apply log Jacobian determinant of transform
  // (var1,var2,rho) -> (W[1,1],W[2,2],W[1,2])

  lp__ <- lp__ + 0.5 * (log(var1) + log(var2));   

  W ~ wishart(4, S);
}
