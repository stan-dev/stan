// geometric lag time-series (Koyck 1951)
//
// http://en.wikipedia.org/wiki/Distributed_lag

data {
  int<lower=0> T;   // number of time points
  real y[T];        // output at time t
  real x[T];        // predictor for time t
}
parameters {
  real alpha;                       // intercept
  real beta;                        // slope
  real <lower=0, upper=1> lambda;   // lag
  real <lower=0> sigma;             // noise scale
}
model {
  alpha ~ cauchy(0,5);
  beta ~ cauchy(0,5);
  lambda ~ uniform(0,1);
  sigma ~ cauchy(0,5);
  for (t in 2:T)
    y[t] ~ normal(alpha +  beta * x[t] +  lambda * y[t-1],
                  sigma);
}

