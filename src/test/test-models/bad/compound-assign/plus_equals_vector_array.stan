generated quantities {
  vector[2] x[3] = {[1,2]', [3,4]', [5,6]'};
  vector[2] y[3] = {[1,2]', [3,4]', [5,6]'};
  x[1] += y[1];
  x += y;
}
