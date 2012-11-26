inv_logit = function(u) { 1.0/(1.0 + exp(-u)); }

sigma.alpha = 0.75;
sigma.beta = 1.5;
J <- 400;  
K <- 100;
alpha <- rnorm(J,0,sigma.alpha);
beta <- rnorm(K,0,sigma.beta);
delta <- 0.75;
gamma <- exp(rnorm(K,0,0.5));
y_all <- matrix(0,nrow=J,ncol=K);
for (j in 1:J)
  for (k in 1:K)
    y_all[j,k] <- rbinom(1,1,inv_logit(gamma[k] * (alpha[j] - beta[k] + delta)));

p_observed = 0.75;
observed <- matrix(rbinom(J*K,1,p_observed),nrow=J,ncol=K);
N <- sum(observed);
y <- rep(-1,N);
jj <- rep(-1,N);
kk <- rep(-1,N);
n <- 1;
for (j in 1:J) {
  for (k in 1:K) {
    if (observed[j,k]) {
      y[n] <- y_all[j,k];
      jj[n] = j;
      kk[n] = k;
      n <- n + 1;
    }
  }
}

dump(c("J","K","N","jj","kk","y"), "irt2_multilevel.data.R");


