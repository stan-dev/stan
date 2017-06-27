generated quantities {
  row_vector[2] x[3] = {[1,2], [3,4], [5,6]};
  row_vector[2] y[3] = {[1,2], [3,4], [5,6]};
  x[1] += y[1];
  x += y;
}
