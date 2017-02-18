transformed data {
  row_vector[2] X;
  X = [ 1, 2];
}
parameters {
  real z;
}
model {
  z ~ normal(0,1);
}
