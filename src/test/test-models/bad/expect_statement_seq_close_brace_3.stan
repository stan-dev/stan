transformed data {
  int vs[2,3];
  int z;
  for (v in vs[1]) {
    z = 3;
  }
  
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
