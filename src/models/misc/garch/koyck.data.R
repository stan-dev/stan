T <- 200;
x <- rnorm(T,0,1);
alpha <- -1.25;
beta <- 0.75;
lambda <- 0.6;
sigma <- 0.5;

y <- rep(NA,T);
y[1] <- rnorm(1,alpha + beta * x[1],sigma);
for (t in 2:T)
  y[t] <- rnorm(1,alpha + beta * x[t] + lambda * y[t-1], sigma);
