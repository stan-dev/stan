library(rstan)
library(ggplot2)
library(reshape2)
unemployment <- read.table ("unemployment.dat", header=TRUE, echo = TRUE)
year <- unemployment$year
y <- unemployment$unemployed.pct

## Plot of the unemployment rate
frame1 = data.frame(year=year,y=y)
p1 <- ggplot(frame1,aes(x=year,y=y)) +
      geom_point() +
      scale_y_continuous("Unemployment") +
      scale_x_continuous("Year") +
      theme_bw()
print(p1)

## Fitting a 1st-order autogregression
## lm(y ~ y_lag)

source("unemployment.data.R", echo = TRUE)    
dataList.1 <- c("N","y_lag","y")
unemployment.sf1 <- stan(file='unemployment.stan', data=dataList.1,
                         iter=1000, chains=4)
print(unemployment.sf1)

## Simulating replicated datasets

lag.post <- extract(unemployment.sf1)
b.hat <- colMeans(lag.post$beta)
s.hat <- mean(lag.post$sigma)

n.sims <- length(year) + 1
n <- 16
y.rep <- array (NA, c(n.sims, n))
for (s in 1:n.sims){
  y.rep[s,1] <- y[1]
  for (t in 2:n){
    prediction <- c (1, y.rep[s,t-1]) %*% b.hat
    y.rep[s,t] <- rnorm (1, prediction, s.hat)
  }
}

## Including uncertainty in the estimated parameters 
n <- 15
n.sims <- length(year)
y.rep2 <- array (NA, c(n.sims, n))
for (s in 1:n.sims){
  for (t in 1:n){
    prediction <-  c (1, y.rep[s+1,t]) %*% lag.post$beta[s,]
    y.rep2[s,t] <- rnorm (1, prediction, lag.post$sigma[s])
  }
}

## Plot of simulated unemployment rate series

y.new <- melt(y.rep2)
y.new$Var2 <- factor(y.new$Var2, levels=c('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'), labels=c('Simulation #1','Simulation #2','Simulation #3','Simulation #4','Simulation #5','Simulation #6','Simulation #7','Simulation #8','Simulation #9','Simulation #10','Simulation #11','Simulation #12','Simulation #13','Simulation #14','Simulation #15'))
frame2 = data.frame(y.new=y.new$value,year=year,Var2=y.new$Var2)
p2 <- ggplot(frame2,aes(y=y.new,x=year)) +
      theme_bw() +
      geom_line() +
      facet_wrap( ~ Var2,ncol=5) +
      theme(axis.title.y = element_blank(),axis.title.x=element_blank())
print(p2)

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
