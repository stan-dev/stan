library("gtools");  # for rdirichlet

V <- 5; # words: river, stream, bank, money, loan

K <- 2; # topics: RIVER, BANK

phi <- array(NA,c(2,5));
phi[1,] = c(0.330, 0.330, 0.330, 0.005, 0.005);
phi[2,] = c(0.005, 0.005, 0.330, 0.330, 0.330);

M <- 25;  # docs
avg_doc_length <- 10;
doc_length <- rpois(M,avg_doc_length);
N <- sum(doc_length);

theta <- rdirichlet(M,alpha);

w <- rep(NA,N);
doc <- rep(NA,N);
n <- 1;
for (m in 1:M) {
  for (i in 1:doc_length[m]) {
    z <- which(rmultinom(1,1,theta[m,]) == 1);
    w[n] <- which(rmultinom(1,1,phi[z,]) == 1);
    doc[n] <- m;
    n <- n + 1;
  }
}

alpha <- rep(1/K,K);
beta <- rep(1/V,V);

dump(c("K","V","M","N","z","w","doc","alpha","beta"),"lda.data.R");
