logit <- function(p) {
  return (log(p / (1-p)));
}

inv_logit <- function(alpha) {
  return (exp(alpha) / (1 + exp(alpha)));
}

logistic_generate_data <- function(N, M, file="dat", seed=0) {
  set.seed(seed);
  alpha <- rnorm(1, sd=1);
  beta <- rnorm(M, sd=1);
  x <- matrix(rnorm(M*N), N, M);
  y <- as.vector(ifelse(inv_logit(alpha + x %*% beta) > runif(N), 1, 0));
  dump(c("N", "M", "y", "x"), file=paste(file, ".Rdata", sep=""));
  dump(c("N", "M", "alpha", "beta", "seed"), file=paste(file, "_param.Rdata", sep=""));
  #return (list(alpha=alpha, beta=beta, x=x, y=y));
}


## generate data
N <- c(1000, 5000, 10000); #, 100000, 1000000);
M <- c(1, 10, 100, 500, 1000);
for (n in N) {
  for (m in M) {
    if (n > m)
      logistic_generate_data(n, m, file=paste("logistic_", n, "_", m, sep=""));
  }
}

