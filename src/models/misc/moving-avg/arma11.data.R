mu <- -1.25;
sigma <- 0.75;
theta <- 0.5;
phi <- 0.2;

T <- 1000;

err <- rnorm(T,0,sigma);
nu <- rep(0,T);
y <- rep(0,T);
y[1] <- err[1] + mu + phi * mu;
for (t in 2:T)
  y[t] <- err[t] + (mu + phi * y[t-1] + theta * err[t-1]);

#library('rstan')
# fit <- stan('arma11.stan', data=list(T=T, y=y), iter=2000, chains=4);
