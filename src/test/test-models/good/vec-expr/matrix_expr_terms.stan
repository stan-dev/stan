transformed data {
  matrix[2,2] X;
  row_vector[2] y;
  y[1] = 10;
  y[2] = 100;

  X = [ [ 1, 2], [ 3, 4] ];
  X = [ y , y ];
}
parameters {
  real z;
}
model {
  z ~ normal(0,1);
}
