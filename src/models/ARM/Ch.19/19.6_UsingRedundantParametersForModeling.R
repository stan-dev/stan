## Read, clean the pilots data, redefine variables
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/schools

library(rstan)
schools <- read.table ("schools.dat", header=TRUE)
attach(schools)

N <- 8
y <- c(28,8,-3,7,-1,1,18,12)
sigma.y <- c(15,10,16,11,9,11,10,18)

dataList.1 <- list(N=N, sigma_y=sigma.y,y=y)
schools.sf1 <- stan(file='schools.stan', data=dataList.1, iter=1000, chains=4)
print(schools.sf1, pars = c("theta", "lp__"))
