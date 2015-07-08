parameters {
  matrix[3,3] a;
  vector[3] b;
}
model {
  matrix[3,3] c;
  c <- a + b;
}
