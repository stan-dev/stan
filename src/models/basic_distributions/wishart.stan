transformed data {
  cov_matrix[2] S;

  for (i in 1:2)
    for (j in 1:2)
      S[i,j] <- 0.0;

  S[1,1] <- 2.0; 
  S[2,2] <- 0.5;
} 
parameters {
  cov_matrix[2] W; 
} 
model {
  W ~ wishart(4, S); 
} 
