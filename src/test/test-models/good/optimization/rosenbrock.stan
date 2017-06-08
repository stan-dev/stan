parameters {
  real x;
  real y;
}

model {
  target += -(pow(1-x,2) + 100*pow(y - pow(x,2),2));
}

