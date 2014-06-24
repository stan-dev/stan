library(rstan)

## Fake-data simulation
a <- 1.4
b <- 2.3
sigma <- 0.9
x <- 1:5
n <- length(x)

# Simulate data, fit the model, and check the coverage of the conf intervals
y <- a + b*x + rnorm (n, 0, sigma)

# (y_x.stan)
# lm(y ~ x)
dataList.1 <- list(N=length(y), y=y, x=x)
y_x.sf1 <- stan(file='y_x.stan', data=dataList.1, iter=1000, chains=4)
print(y_x.sf1)

post <- extract(y_x.sf1)
b.hat <- colMeans(post$beta)[2] # "b" is the 2nd coef in the model
b.se <- sd(post$beta[,2]) / sqrt(4000)     # "b" is the 2nd coef in the model

cover.68 <- abs (b - b.hat) < b.se     # this will be TRUE or FALSE
cover.95 <- abs (b - b.hat) < 2*b.se   # this will be TRUE or FALSE
cat (paste ("68% coverage: ", cover.68, "\n"))
cat (paste ("95% coverage: ", cover.95, "\n"))

# Put it in a loop

n.fake <- 1000
cover.68 <- rep (NA, n.fake)
cover.95 <- rep (NA, n.fake)
for (s in 1:n.fake){
  y <- a + b*x + rnorm (n, 0, sigma)
  dataList.1 <- list(N=length(y), y=y, x=x)
  y_x.sf1 <- sampling(y_x.sm, dataList.1)
  print(y_x.sf1)
  post <- extract(y_x.sf1)
  b.hat <- colMeans(post$beta)[2] # "b" is the 2nd coef in the model
  b.se <- sd(post$beta[,2]) / sqrt(4000)     # "b" is the 2nd coef in the model
  cover.68[s] <- abs (b - b.hat) < b.se  
  cover.95[s] <- abs (b - b.hat) < 2*b.se 
}
cat (paste ("68% coverage: ", mean(cover.68), "\n"))
cat (paste ("95% coverage: ", mean(cover.95), "\n"))

# Do it again, this time using t intervals

n.fake <- 1000
cover.68 <- rep (NA, n.fake)
cover.95 <- rep (NA, n.fake)
t.68 <-  qt (.84, n-2)
t.95 <-  qt (.975, n-2)
for (s in 1:n.fake){
  y <- a + b*x + rnorm (n, 0, sigma)
  dataList.1 <- list(N=length(y), y=y, x=x)
  y_x.sf1 <- sampling(y_x.sm, dataList.1)
  print(y_x.sf1)
  post <- extract(y_x.sf1)
  b.hat <- colMeans(post$beta)[2] # "b" is the 2nd coef in the model
  b.se <- sd(post$beta[,2]) / sqrt(4000)     # "b" is the 2nd coef in the model
  cover.68[s] <- abs (b - b.hat) < t.68*b.se  
  cover.95[s] <- abs (b - b.hat) < t.95*b.se 
}
cat (paste ("68% coverage: ", mean(cover.68), "\n"))
cat (paste ("95% coverage: ", mean(cover.95), "\n"))  
