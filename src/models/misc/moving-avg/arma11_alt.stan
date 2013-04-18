# ARMA(1,1) with Wayne Folta-style err encoding

data {
  int<lower=1> T;       // number of observations
  real y[T];            // observed outputs
}
parameters {
  real mu;              // mean term
  real phi;             // autoregression coeff
  real theta;           // moving avg coeff
  real<lower=0> sigma;  // noise scale
}
model {
  real err;
  mu ~ normal(0,10);
  phi ~ normal(0,2);
  theta ~ normal(0,2);
  sigma ~ cauchy(0,5);
  err <- y[1] - mu + phi * mu;
  err ~ normal(0,sigma);
  for (t in 2:T) {
    err <- y[t] - (mu + phi * y[t-1] + theta * err); 
    err ~ normal(0,sigma);
  }
}
