data {
  array[4] int a;
  array[3] int b;
}
transformed data {
  int c;
  c = a[1] / b[2];
}
model {

}
