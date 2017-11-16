// test right value passed through for, foreach, while and embedded
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

    // nested foreach (array)
    while (1) {
      int vs[2];
      fora (v in vs) break; 
      //FOREACHCHANGE: this needs to be changed (range)
      fora (v in vs) continue;
      //FOREACHCHANGE: this needs to be changed (range)
    }

    // nested foreach (matrix)
    while (1) {
      matrix[2,3] vs;
      form (v in vs) break; 
      //FOREACHCHANGE: this needs to be changed (range)
      form (v in vs) continue;
      //FOREACHCHANGE: this needs to be changed (range)
      form (v in vs) v = 3;
    }

    // nested foreach (vector)
    while (1) {
      vector[2] vs;
      form (v in vs) break; 
      //FOREACHCHANGE: this needs to be changed (range)
      form (v in vs) continue;
      //FOREACHCHANGE: this needs to be changed (range)
      form (v in vs) v = 3;
    }

    // nested foreach (rowvector)
    while (1) {
      row_vector[2] vs;
      form (v in vs) break; 
      //FOREACHCHANGE: this needs to be changed (range)
      form (v in vs) continue;
      //FOREACHCHANGE: this needs to be changed (range)
      form (v in vs) v = 3;
    }

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
