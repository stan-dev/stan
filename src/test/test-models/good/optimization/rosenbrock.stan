parameters {
  real x;
  real y;
}

model {
  increment_log_prob( -(pow(1-x,2) + 100*pow(y - pow(x,2),2)) );
}

