transformed data {
    matrix(10,20) A[17];
    matrix(3,4) B[5];

    B[1,2,3] <- 2.0;
}
model {
    B[1,2,3] ~ normal(0,1);    
}
