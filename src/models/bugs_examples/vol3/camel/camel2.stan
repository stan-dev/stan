# Camel: Multivariate normal with structured missing data 
# http://www.openbugs.info/Examples/Camel.html


## integrate out the missings.  

transformed data {

  vector[2] Y[4]; 
  real Y1[4];  // missing y2 
  real Y2[4];  // msising y1

  vector[2] mu; 
  matrix[2, 2] S; 

  mu[1] <- 0; 
  mu[2] <- 0; 
  S[1, 1] <- 1000; 
  S[1, 2] <- 0;
  S[2, 1] <- 0;
  S[2, 2] <- 1000;

  Y[1, 1] <- 1.; 
  Y[1, 2] <- 1.; 
  Y[2, 1] <- 1.; 
  Y[2, 2] <- -1.; 
  Y[3, 1] <- -1.; 
  Y[3, 2] <- 1.; 
  Y[4, 1] <- -1.; 
  Y[4, 2] <- -1.; 

  Y1[1] <- 2.; 
  Y1[2] <- 2.; 
  Y1[3] <- -2.; 
  Y1[4] <- -2.; 
  
  Y2[1] <- 2.; 
  Y2[2] <- 2.; 
  Y2[3] <- -2.; 
  Y2[4] <- -2.; 
} 

parameters { 
  cov_matrix[2] Sigma; 
} 

transformed parameters {
  real<lower=-1,upper= 1> rho; 
  rho <- Sigma[1, 2] / sqrt(Sigma[1, 1] * Sigma[2, 2]); 
} 


model {
  for (n in 1:4) Y[n] ~ multi_normal(mu, Sigma); 
  Y1 ~ normal(0, sqrt(Sigma[1, 1]));
  Y2 ~ normal(0, sqrt(Sigma[2, 2])); 
  // Sigma ~ inv_wishart(2, S); 
  lp__ <- lp__ - 1.5 * log(determinant(Sigma)); 
} 

