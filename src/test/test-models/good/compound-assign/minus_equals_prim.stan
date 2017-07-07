functions {
  real foo(real a1) {
    real b;
    int c;
    b -= a1;
    c -= c;
    return b;
  }
}
data {
  int b;
  int c;
}
transformed data {
  int x = 10;
  real y = 20;
  int ax[3] = { 1, 2, 3 };
  real ay[3] = { 1.0, 2.0, 3.0 };
  x -= 1;
  x -= c;
  x -= ax[1];
  y -= 1;
  y -= 1.0;
  y -= b;
  y -= c;
  y -= ax[1];
  y -= ay[1];
  y -= foo(y);
}
transformed parameters {
  real w = 30;
  w -= b;
  w -= c;
  w -= x;
  w -= y;
  w -= ax[1];
  w -= ay[1];
  w -= foo(w);
}  
model {
  real v = 7;
  v -= b;
  v -= c;
  v -= ax[1];
  v -= ay[1];
  v -= y;
  v -= foo(y);
  v -= w;
  v -= foo(w);
  v -= v;
  v -= foo(v);
}
generated quantities {
  real z = 40;
  z -= b;
  z -= c;
  z -= ax[1];
  z -= ay[1];
  z -= w;
  z -= foo(w);
  z -= y;
  z -= foo(y);
}
