
T <- 1000;
I <- 4;
K <- 2;
sigma0 <- 1;
sigma <- rep(sigma0,I);
F_sim <- matrix(c(0.2, 0.3, 0.1, 0.4, 0.8, 0.1, 0.1, 0), nrow = 2, ncol=4, byrow=TRUE);
G_sim <- matrix(exp(rnorm(T*K,2,1)), nrow=T, ncol=K);
X <- G_sim %*% F_sim + abs(rnorm(T*I,0,sigma0));
