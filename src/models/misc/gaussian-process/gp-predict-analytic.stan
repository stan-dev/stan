// Predict from Gaussian Process w. analytic posterior
// Fixed covar function: eta_sq=1, rho_sq=1, sigma_sq=0.1
data {
  int<lower=1> N1;     
  vector[N1] x1; 
  vector[N1] y1;
  int<lower=1> N2;
  vector[N2] x2;
}
transformed data {
  vector[N2] mu;
  matrix[N2,N2] L;
  { 
    matrix[N1,N1] Sigma;
    matrix[N2,N2] Omega;
    matrix[N1,N2] K;
    
    matrix[N2,N1] K_transpose_div_Sigma;
    matrix[N2,N2] Tau;

    for (i in 1:N1) 
      for (j in 1:N1)
        Sigma[i,j] <- exp(-pow(x1[i] - x1[j],2)) 
          + if_else(i==j, 0.1, 0.0);
    for (i in 1:N2) 
      for (j in 1:N2)
        Omega[i,j] <- exp(-pow(x2[i] - x2[j],2)) 
          + if_else(i==j, 0.1, 0.0); 
    for (i in 1:N1)
      for (j in 1:N2)
        K[i,j] <- exp(-pow(x1[i] - x2[j],2));
    
    K_transpose_div_Sigma <- K' / Sigma;
    mu <- K_transpose_div_Sigma * y1; 
    Tau <- Omega - K_transpose_div_Sigma * K;
    for (i in 1:N2)
      for (j in (i+1):N2)
        Tau[i,j] <- Tau[j,i];

    L <- cholesky_decompose(Tau);
  }
}
parameters {
  vector[N2] z;
}
model {
  z ~ normal(0,1);
}
generated quantities {
  vector[N2] y2;    // y2 ~ multi_normal(mu, Tau);
  y2 <- mu + L * z;

}
