transformed data {
  vector[5] v = [1, 2, 3, 4, 5]';
  real y = 0.1;

  print("y: ", y);
  y += 1;
  print("expect 1.1 actual y: ", y);
  
  print("v: ", v);
  v[1] = v[2];
  print("expect: [2,2,3,4,5]  v: ", v);
  v[1:2] += 1.0;
  print("expect: [3,3,3,4,5]  v: ", v);
  v[2:4] += v[1:3] + 1;
  print("expect: [3,7,7,8,5]  v: ", v);
}
