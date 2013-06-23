skewed_simplex <- function(k) {
  result <- (1/(2:(1+k)))^2;
  result <- result / sum(result);
  return(result);
}

K <- 4;
V <- 10;
theta <- skewed_simplex(K);
phi <- matrix(NA,nrow=K,ncol=V)
for (k in 1:K)
  phi[k,] <- permute(skewed_simplex(V));

M <- 200;  # docs
avg_doc_length <- 10;
doc_length <- rpois(M,avg_doc_length);
N <- sum(doc_length);

z <- rep(NA,M);
w <- rep(NA,N);
doc <- rep(NA,N);
n <- 1;
for (m in 1:M) {
  z[m] <- which(rmultinom(1,1,theta) == 1);
  for (i in 1:doc_length[m]) {
    w[n] <- which(rmultinom(1,1,phi[z[m],]) == 1);
    doc[n] <- m;
    n <- n + 1;
  }
}
alpha <- rep(1,K);
beta <- rep(0.1,V);  # prior count < 1 for words

dump(c("K","V","M","N","z","w","doc","alpha","beta"),file="naive-bayes.data.R");


