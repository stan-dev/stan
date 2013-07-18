library(rstan)
library(ggplot2)

unemployment <- read.table ("unemployment.dat", header=TRUE)
year <- unemployment$year
y <- unemployment$unemployed.pct

## Plot of the unemployment rate
frame1 = data.frame(year=year,y=y)
m2 <- ggplot(frame1,aes(x=year,y=y))
m2 + geom_point() + scale_y_continuous("Unemployment") + scale_x_continuous("Year") + theme_bw()

## Fitting a 1st-order autogregression
if (!file.exists("unemployment.sm.RData")) {
    rt <- stanc("unemployment.stan", model_name="unemployment")
    unemployment.sm <- stan_model(stanc_ret=rt)
    save(unemployment.sm, file="unemployment.sm.RData")
} else {
    load("unemployment.sm.RData", verbose=TRUE)
}
source("unemployment.data.R")    
dataList.1 <- list(N=N, y_lag=y_lag,y=y)
unemployment.sf1 <- sampling(unemployment.sm, dataList.1)
print(unemployment.sf1)

## Simulating replicated datasets

beta.post <- extract(unemployment.sf1, "beta")$beta
sigma.post <- extract(unemployment.sf1, "sigma")$sigma
b.hat <- colMeans(beta.post)
s.hat <- mean(sigma.post)

n.sims <- 1000
y.rep <- array (NA, c(n.sims, n))
for (s in 1:n.sims){
  y.rep[s,1] <- y[1]
  for (t in 2:n){
    prediction <- c (1, y.rep[s,t-1]) %*% b.hat
    y.rep[s,t] <- rnorm (1, prediction, s.hat)
  }
}

## Including uncertainty in the estimated parameters

lm.lag <- lm (y ~ y_lag)
lm.lag.sim <- sim (lm.lag, n.sims)       # simulations of beta and sigma
for (s in 1:n.sims){
  y.rep[s,1] <- y[1]
  for (t in 2:n){
    prediction <-  c (1, y.rep[s,t-1]) %*% lm.lag.sim@coef[s,]
    y.rep[s,t] <- rnorm (1, prediction, lm.lag.sim@sigma[s])
  }
}

## Plot of simulated unemployment rate series FIXME:won't compile. mismathc dimensions

for (s in 1:15){
  frame2 = data.frame(year=year,y=y.rep[s,])
m2 <- ggplot(frame1,aes(x=year,y=y))
m2 + geom_point() + scale_y_continuous("Unemployment") + scale_x_continuous("Year") + theme_bw() + labs(title=paste("simulation #",s,sep=""))
}

## Numerical model check

Test <- function (y){
  n <- length (y)
  y.lag <- c (NA, y[1:(n-1)])
  y.lag2 <- c (NA, NA, y[1:(n-2)])
  sum (sign(y-y.lag) != sign(y.lag-y.lag2), na.rm=TRUE)
}

n.sims <- 1000
print (Test (y))
test.rep <- rep (NA, n.sims)
for (s in 1:n.sims){
  test.rep[s] <- Test (y.rep[s,])
}

print (mean (test.rep > Test(y)))
print (quantile (test.rep, c(.05,.5,.95)))
