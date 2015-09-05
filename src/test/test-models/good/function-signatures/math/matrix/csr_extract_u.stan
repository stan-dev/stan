data { 
  matrix[3, 4] a_d;
}
transformed data {
  int u[5];
  u <- csr_extract_u(a_d);
}
parameters {
  real y;
  matrix[3, 4] a_p;
}
transformed parameters {
  real v;
  { 
    int u_p[5];
    u_p <- csr_extract_u(a_d);
    u_p <- csr_extract_u(a_p);
    v <- 3;
  }
}
model {  
  y ~ normal(0, 1);
}


