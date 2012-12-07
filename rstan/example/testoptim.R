library(rstan)
stdnorm <- '
data {
  int N;
  real y[N];
}

parameters {
  real mu;
  real<lower=0> sigma;
}

model {
  y ~ normal(mu, sigma);
}
'

N <- 30
y <- rnorm(N)
cat("mean(y)=", mean(y), ", and sd(y)=", sd(y), "\n", sep = '')
dat <- list(N = N, y = y)
sm <- stan_model(model_code = stdnorm)

optim <- rstan:::optimizing(sm, data = dat)
print(optim)

