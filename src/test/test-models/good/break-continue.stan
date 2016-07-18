// test right value passed through for, while and embedded
functions {
  int foo(int a) {
    // direct while
    while (1) break;
    while (0) continue;

    // direct for
    for (i in 1:10) break;
    for (i in 1:10) continue;

    // in statement seq
    while (1) {
      int b;
      b = 5;
      break;
    }
    
    // if, else if, else body
    while (1) {
      if (0) break;
      else if (1) break;
      else break;
    }

    // nested while
    while (1) while (0) break;

    // nested for
    while (1) for (i in 1:10) break;

    // nested block
    while (1) {
      int b;
      b = 5;
      {
        int c;
        c = 6;
        break;
      }
    }

    return 0;
  }
}
transformed data {
  int x;
  x = 0;
  while (0) break;
  while (1) continue;
}
parameters {
  real y;
}
transformed parameters {
  real z;
  z = 1;
  while (0) break;
  while (1) continue;
}
model {
}
generated quantities {
   real u;
   u = 1;
   while (1) break;
   while (0) continue;
}
