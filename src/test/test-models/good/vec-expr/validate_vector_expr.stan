functions {
}
data { 
  int d;
} 
transformed data {
  vector[3] td_v1 = [4.4, 5.0^2, d]';
  row_vector[3] td_rv1 = [1.0, 2.0^2, d];
}
parameters {
  real y;
}
transformed parameters {
  row_vector[3] tp_rv1 = [1.0, y, d];
}
model {
  y ~ normal(0, 1);
}
generated quantities {
  vector[3] gq_v1 = [7.0, 8^2, d]';
}
