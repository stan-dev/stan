parameters {
  matrix[3,3] a;
}
model {
  int b[4];
  matrix[3,3] c;
  c <- b % a;
}
