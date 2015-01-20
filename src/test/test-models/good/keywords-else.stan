parameters {
  real a;
  real elseif;
}
model {
  if (1 > 2) {
    a ~ normal(0,1);
  } 
  elseif ~ normal(0,1);
}
