
phi <- 0.95;
sigma <- 0.25;
beta <- 0.6;
mu <- 2 * log(beta);

T <- 500;

h <- rep(NA,T);
h[1] <- rnorm(1, mu, sigma / sqrt(1 - phi * phi));
for (t in 2:T)
  h[t] <- rnorm(1, mu + phi * (h[t-1] - mu), sigma);
y <- rep(NA,T);
for (t in 1:T)
  y[t] <- rnorm(1, 0, exp(h[t] / 2));
