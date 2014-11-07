data {
  real<lower=1.2> a;
  real<lower=1> b;
  real<lower=1.2,upper=2> c;  
  real<lower=1,upper=1.2> d;  
  real<lower=1.1,upper=1.2> e;
  real<lower=1,upper=2> f;
}
parameters {
  real y;
}
model {
  y ~ normal(0,1);
}
