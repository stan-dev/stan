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
  vector[T] nu;         // prediction for time t
  vector[T] err;        // error for time t
  nu[1] <- mu + phi * mu;   // assume err[0] == 0
  err[1] <- y[1] - nu[1];
  for (t in 2:T) {
    nu[t] <- mu + phi * y[t-1] + theta * err[t-1];
    err[t] <- y[t] - nu[t];
  }

  // priors
  mu ~ normal(0,10);
  phi ~ normal(0,2);
  theta ~ normal(0,2);
  sigma ~ cauchy(0,5);

  // likelihood
  err ~ normal(0,sigma);
}

// alternative encoding:
//
// model {
//   err <- 0;
//   for (t in 2:T) {
//     err <- y[t-1] - (mu + phi * y[t-1] + theta * err[t-1]);
//     err ~ normal(0,sigma);
//   }

