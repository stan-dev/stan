library(rstan)
library(ggplot2)
source("lightspeed.data.R")    

## Model fit (lightspeed.stan)
## lm (y ~ 1)
if (!file.exists("lightspeed.sm.RData")) {
    rt <- stanc("lightspeed.stan", model_name="lightspeed")
    lightspeed.sm <- stan_model(stanc_ret=rt)
    save(lightspeed.sm, file="lightspeed.sm.RData")
} else {
    load("lightspeed.sm.RData", verbose=TRUE)
}

dataList.1 <- list(N=N, y=y)
lightspeed.sf1 <- sampling(lightspeed.sm, dataList.1)
print(lightspeed.sf1)

## Create the replicated data 

light <- lm (y ~ 1)
n.sims <- 1000
sim.light <- sim (light, n.sims)

## Create fake data 

n <- length (y)
y.rep <- array (NA, c(n.sims, n))
for (s in 1:n.sims){
  y.rep[s,] <- rnorm (n, coef(sim.light)[s], slot(sim.light,"sigma")[s])
}

## Histogram of replicated data (Figure 8.4)

for (s in 1:20){
  frame = data.frame(x1=y.rep[s,])
  m <- ggplot(frame,aes(x=x1)) +  geom_histogram(colour = "black", fill = "white",binwidth=5) + theme_bw()
  print(m)
}

## Write a function to make histograms with specified bin widths and ranges

Hist.preset <- function (a, width, xtitle,ytitle,maintitle){
  a.hi <- max (a, na.rm=TRUE)
  a.lo <- min (a, na.rm=TRUE)
  if (is.null(width)) width <- min (sqrt(a.hi-a.lo), 1e-5)
  bin.hi <- width*ceiling(a.hi/width)
  bin.lo <- width*floor(a.lo/width)
  frame1 = data.frame(x1=a)
  m1 <- ggplot(frame,aes(x=x1)) +  geom_histogram(colour = "black", fill = "white", binwidth=width) + theme_bw() + scale_x_continuous(xtitle) + scale_y_continuous(ytitle) + labs(title=maintitle)
  print(m1)
}

## Run the function

for (s in 1:20){
  Hist.preset (y.rep[s,], width=5, "","",paste("Replication #",s,sep=""))
}

## Numerical test

Test <- function (y){
  min (y)
}
test.rep <- rep (NA, n.sims)
for (s in 1:n.sims){
  test.rep[s] <- Test (y.rep[s,])
}

## Histogram Figure 8.5

  frame2 = data.frame(x1=test.rep)
  m2 <- ggplot(frame,aes(x=x1)) +  geom_histogram(colour = "black", fill = "white") + theme_bw() + labs(title="Observed T(y) and distribution of T(y.rep)")
  print(m2)


hist (test.rep, xlim=range (Test(y), test.rep), yaxt="n", ylab="",
 xlab="", main="Observed T(y) and distribution of T(y.rep)")
lines (rep (Test(y), 2), c(0,10*n))

##FIXME:ADD ROACH STUFF.. currently not correct
