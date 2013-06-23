K <- 3;

N1 <- 500;
N2 <- 300;
N3 <- 200;
N <- N1 + N2 + N3;
mu1 <- -2.0;
mu2 <- 0.5;
mu3 <- 3.5;
sigma1 <- 1.0;
sigma2 <- 0.5;
sigma3 <- 2;

y <- c(rnorm(N1,mu1,sigma1),rnorm(N2,mu2,sigma2),rnorm(N3,mu3,sigma3));
