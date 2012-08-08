transformed data {
  cov_matrix[3] S;
  S[1, 1] <- 2; S[1, 2] <- 0; S[1, 3] <- 0; 
  S[2, 2] <- 1; S[2, 1] <- 0; S[2, 3] <- 0;
  S[3, 3] <- .5; S[3, 1] <- 0; S[3, 2] <- 0; 
} 
parameters {
  cov_matrix[3] W; 
} 
model {
  W ~ inv_wishart(5, S); 
} 
