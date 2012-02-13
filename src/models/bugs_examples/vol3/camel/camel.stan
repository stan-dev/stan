# Camel: Multivariate normal with structured missing data 
# http://www.openbugs.info/Examples/Camel.html

#   data {
#     int(0,) N; # N = 12 
#     
#   } 

transformed data {
  vector(2) mu; 
  matrix(2, 2) S; 
  mu[1] <- 0; 
  mu[2] <- 0; 
  S[1, 1] <- 1000; 
  S[1, 2] <- 0;
  S[2, 1] <- 0;
  S[2, 2] <- 1000;
} 

parameters { 
  cov_matrix(2) Sigma; 
  double y52; 
  double y62; 
  double y72; 
  double y82; 
  double y91; 
  double y101; 
  double y111; 
  double y121; 
} 

transformed parameters {
  vector(2) Y[12]; 
  Y[1, 1] <- 1; 
  Y[1, 2] <- 1; 
  Y[2, 1] <- 1; 
  Y[2, 2] <- -1; 
  Y[3, 1] <- -1; 
  Y[3, 2] <- 1; 
  Y[4, 1] <- -1; 
  Y[4, 2] <- -1; 

  Y[5, 1] <- 2; 
  Y[6, 1] <- 2; 
  Y[7, 1] <- -2; 
  Y[8, 1] <- -2; 
  Y[5, 2] <- y52; 
  Y[6, 2] <- y62; 
  Y[7, 2] <- y72; 
  Y[8, 2] <- y82; 
  Y[9, 1] <- y91; 
  Y[10, 1] <- y101; 
  Y[11, 1] <- y111; 
  Y[12, 1] <- y121; 
  Y[9, 2] <- 2; 
  Y[10, 2] <- 2; 
  Y[11, 2] <- -2; 
  Y[12, 2] <- -2; 
} 


model {
  Sigma ~ inv_wishart(2, S); 
  for (n in 1:12) Y[n] ~ multi_normal(mu, Sigma); 
} 

generated quantities { 
  double(-1, 1) rho; 
  rho <- Sigma[1, 2] / sqrt(Sigma[1, 1] * Sigma[2, 2]); 
} 
