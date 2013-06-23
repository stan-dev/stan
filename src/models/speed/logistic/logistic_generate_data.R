logit <- function(p) {
  return (log(p / (1-p)));
}

inv_logit <- function(alpha) {
  return (1 / (1 + exp(-alpha)));
}

logistic_generate_data <- function(N, M, file="dat", seed=0) {
  set.seed(seed);
  beta <- rnorm(M, sd=1);
  x <- matrix(0,N,M);
  x[1:N,2:M] <- matrix(rnorm((M-1)*N), N, M-1);  # same scale (favors Stan, which is not scale invariant)
  x[1:N,1] <- 1
  y <- as.vector(ifelse(inv_logit(x %*% beta) > runif(N), 1, 0));
  dump(c("N", "M", "y", "x"), file=paste(file, ".data.R", sep=""));
  dump(c("beta", "seed"), file=paste(file, "_param.data.R", sep=""));
  #return (list(alpha=alpha, beta=beta, x=x, y=y));
}


## generate data
N <- c(128, 1024, 4096); #, 16K-ish
M <- c(2, 8, 32, 128, 512); # , 2048);
for (n in N) {
  for (m in M) {
    if (n > m)
      logistic_generate_data(n, m, file=paste("logistic_", n, "_", m, sep=""));
  }
}

