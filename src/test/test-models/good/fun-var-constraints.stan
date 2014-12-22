functions {
  real foo() {
    return 0;
  }
}
data {
  real<lower=foo(),upper=1> b;
}
transformed data {
  real<lower=-100,upper=foo()> bt;
}
parameters {
  real<lower=foo(),upper=1> y;
}
transformed parameters {
  real<lower=-100,upper=foo()> yt;
}
model {
  y ~ normal(0,1);
}  
generated quantities {
  real<lower=-100,upper=foo()> g;
}
