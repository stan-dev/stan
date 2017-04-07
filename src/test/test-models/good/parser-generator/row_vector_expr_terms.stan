data {
  real x;
  real y;
}
transformed data {
  row_vector[2] X = [ 1, 2];
  X = [ x, y];
  X = [ x + y, x - y];
  X = [ x^2, y^2];
}
parameters {
  real z;
}
transformed parameters {
  vector[3] WT = [ 1, 1, 1]';
  row_vector[2] Z = [ 1, z];
  Z = [ x, y];
}
model {
  z ~ normal(0,1);
}
