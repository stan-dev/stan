data { 
  matrix[3, 4] a_d;
}
transformed data {
  int v_d[5];
  v_d <- csr_extract_v(a_d);
}
parameters {
  real y_p;
  matrix[3, 4] a_p;
}
transformed parameters {
  real v;
  { 
    int v_p[5];
    v_p <- csr_extract_v(a_d);
    v_p <- csr_extract_v(a_p);
    v <- 3;
  }
}
model {  
  y_p ~ normal(0, 1);
}
