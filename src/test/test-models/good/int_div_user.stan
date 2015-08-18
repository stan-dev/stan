data {
  int a[4];
  int b[3];
}
transformed data {
  int c;
  c <- a[1] / b[2];
}
model {
}
