data { 
  matrix[3,4] a_d;
}
transformed data {
  vector[3] w_d;
  w_d <- csr_extract_w(a_d);
}
parameters {
  real y;
  matrix[3,4] a_p;
}
transformed parameters {
  vector[3] w_p;
  w_p <- csr_extract_w(a_d);
  w_p <- csr_extract_w(a_p);
}
model {  
  y ~ normal(0, 1);
}
