functions {
  int foo(int a) {
    while (1) break;
    while (0) continue;
    for (i in 1 : 10) 
      break;
    for (i in 1 : 10) 
      continue;
    while (1) {
      int b;
      b = 5;
      break;
    }
    while (1) {
      if (0) 
        break;
      else if (1) 
        break;
      else 
        break;
    }
    while (1) while (0) break;
    while (1) {
      for (i in 1 : 10) 
        break;
    }
    while (1) {
      array[2, 3] int vs;
      int z;
      for (v in vs) {
        z = 0;
        break;
      }
      for (v in vs) {
        z = 0;
        continue;
      }
      for (v in vs) {
        for (vv in v) {
          z = 0;
          break;
        }
        z = 1;
      }
    }
    while (1) {
      real z;
      matrix[2, 3] vs;
      for (v in vs) {
        z = 0;
        break;
      }
      for (v in vs) {
        z = 3.2;
        continue;
      }
    }
    while (1) {
      real z;
      vector[2] vs;
      for (v in vs) {
        z = 0;
        break;
      }
      for (v in vs) {
        z = 3.2;
        continue;
      }
    }
    while (1) {
      real z;
      row_vector[2] vs;
      for (v in vs) {
        z = 0;
        break;
      }
      for (v in vs) {
        z = 3.2;
        continue;
      }
    }
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

