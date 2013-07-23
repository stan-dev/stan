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

n.sims <- length(year)
n <- length (year)
y.rep <- array (NA, c(n.sims, n))
for (s in 1:n.sims){
  y.rep[s,1] <- y[1]
  for (t in 2:n){
    prediction <- c (1, y.rep[s,t-1]) %*% b.hat
    y.rep[s,t] <- rnorm (1, prediction, s.hat)
  }
}

## Including uncertainty in the estimated parameters (y_x.stan)
## lm(y ~ y_lag)
if (!file.exists("y_x.sm.RData")) {
    rt <- stanc("y_x.stan", model_name="y_x")
    y_x.sm <- stan_model(stanc_ret=rt)
    save(y_x.sm, file="y_x.sm.RData")
} else {
    load("y_x.sm.RData", verbose=TRUE)
}
dataList.1 <- list(N=length(y), y=y, x=y_lag)
y_x.sf1 <- sampling(grades.sm, dataList.1)
print(y_x.sf1)
lag.post <- extract(y_x.sf1)

n.sims <- length(year)
for (s in 1:n.sims){
  y.rep2[s,1] <- y[1]
  for (t in 2:n){
    prediction <-  c (1, y.rep[s,t-1]) %*% lag.post$beta[s,]
    y.rep2[s,t] <- rnorm (1, prediction, lag.post$sigma[s])
  }
}

## Plot of simulated unemployment rate series FIXME:won't compile. mismathc dimensions

for (s in 1:15){
  frame2 = data.frame(year=year,y2=y.rep[s,])
  m2 <- ggplot(frame2,aes(x=year,y=y2))
  m2 + geom_point() + scale_y_continuous("Unemployment") + scale_x_continuous("Year") + theme_bw() + labs(title=paste("simulation #",s,sep=""))
}

unemployment15 <- data.frame(year=year, y1=y.rep)
ss <- 1:15
dev.new()
p <- ggplot(data=unemployment15, aes(x=year, y=y1)) + scale_x_continuous(breaks=c(0,1), labels=c("0", "1")) + factor(ss, levels=ss[matrix(ss, nrow=3, byrow=FALSE)]) + facet_wrap(~y1, ncol = 5)
print(p)










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
