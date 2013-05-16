require(MCMCpack)

# CONSTANTS
K <- 3;
V <- 10;
T <- 100;
T_unsup <- 500;
alpha <- rep(1,K);
beta <- rep(0.1,V);

# DATA
w <- rep(0,T);
z <- rep(0,T);
u <- rep(0,T_unsup);

# PARAMETERS
theta <- rdirichlet(K,alpha);
phi <- rdirichlet(K,beta);

# SIMULATE DATA

# supervised
z[1] <- sample(1:K,1);
for (t in 2:T)
  z[t] <- sample(1:K,1,replace=TRUE,theta[z[t - 1], 1:K]);
for (t in 1:T)
  w[t] <- sample(1:V,1,replace=TRUE,phi[z[t],1:V]);

# unsupervised
y <- rep(0,T_unsup);  
y[1] <- sample(1:K,1);
for (t in 2:T_unsup)
  y[t] <- sample(1:K,1,replace=TRUE,theta[y[t-1],1:K]);
for (t in 1:T_unsup)
  u[t] <- sample(1:V,1,replace=TRUE,phi[y[t], 1:V]);
