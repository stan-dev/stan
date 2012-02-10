transformed data {
  cov_matrix(3) S;

  for (i in 1:3)
    for (j in 1:3)
      S[i,j] <- 0.0;

  S[1,1] <- 2.0; 
  S[2,2] <- 1.0; 
  S[3,3] <- 0.5;
} 
parameters {
  cov_matrix(3) W; 
} 
model {
  W ~ wishart(5, S); 
} 
