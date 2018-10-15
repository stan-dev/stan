transformed data {
  matrix[2, 3] a = [ [1, 2, 3], [4, 5, 6] ];
  a /= 5.0;
  print("a: ", a);
  a[1, ] /= 5.0;
  print("r1 div 5: ", a);
}
