parameters {
  real y;
}
model {
  real loc_real_dim2[2,1] = {{},{}}; // cannot be empty
  y ~ normal(0,1);
}
