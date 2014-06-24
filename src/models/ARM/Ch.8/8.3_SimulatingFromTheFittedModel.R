library(rstan)
library(ggplot2)
library(reshape2)
source("lightspeed.data.R", echo = TRUE)    

## Model fit (lightspeed.stan)
## lm (y ~ 1)
dataList.1 <- c("N","y")
lightspeed.sf1 <- stan(file='lightspeed.stan', data=dataList.1,
                       iter=1000, chains=4)
print(lightspeed.sf1)
post <- extract(lightspeed.sf1)

## Create the replicated data 

n.sims <- 1000

## Create fake data 

n <- 15
y.rep <- array (NA, c(n.sims, n))
for (s in 1:n.sims){
  y.rep[s,] <- rnorm (n, post$beta[s], post$sigma[s])
}

## Histogram of replicated data (Figure 8.4)
y.new <- melt(y.rep)
y.new$Var2 <- factor(y.new$Var2, levels=c('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'), labels=c('Replication #1','Replication #2','Replication #3','Replication #4','Replication #5','Replication #6','Replication #7','Replication #8','Replication #9','Replication #10','Replication #11','Replication #12','Replication #13','Replication #14','Replication #15'))
p1 <- ggplot(y.new,aes(value)) +
      geom_histogram(colour = "black", fill = "white",binwidth=5) +
      theme_bw() +
      facet_wrap( ~ Var2,ncol=5) +
      theme(axis.title.y = element_blank(),axis.title.x=element_blank())
print(p1)

## Write a function to make histograms with specified bin widths and ranges

Hist.preset <- function (a, width, xtitle,ytitle,maintitle){
  dev.new()
  a.hi <- max (a, na.rm=TRUE)
  a.lo <- min (a, na.rm=TRUE)
  if (is.null(width)) width <- min (sqrt(a.hi-a.lo), 1e-5)
  bin.hi <- width*ceiling(a.hi/width)
  bin.lo <- width*floor(a.lo/width)
  frame1 = data.frame(x1=a)
  p2 <- ggplot(frame,aes(x=x1)) +
        geom_histogram(colour = "black", fill = "white", binwidth=width) +
        theme_bw() +
        scale_x_continuous(xtitle) +
        scale_y_continuous(ytitle) +
        labs(title=maintitle)
  print(p2)
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
dev.new()
  frame2 = data.frame(x1=test.rep)
  p3 <- ggplot(frame,aes(x=x1)) +
        geom_histogram(colour = "black", fill = "white") +
        theme_bw() +
        labs(title="Observed T(y) and distribution of T(y.rep)")
  print(p3)

##############################################################################
## Read the cleaned data
# All data are at http://www.stat.columbia.edu/~gelman/arm/examples/roaches
# if bad initial values, this model fails
# NOTE: can't find same exact data set as ARM book uses..

roachdata <- read.csv ("roachdata.csv")
attach(roachdata)

dataList.1 <- list(N=length(roachdata$y), y=roachdata$y,roach1=roachdata$roach1,
                   treatment=roachdata$treatment,exposure2=roachdata$exposure2,
                   senior=roachdata$senior)
roaches.sf1 <- stan(file='roaches.stan', data=dataList.1, iter=5000, chains=4)
print(roaches.sf1)
post <- extract(roaches.sf1)

## Comparing the data to a replicated dataset

n <- length(y)
X <- cbind (rep(1,n), roach1, treatment, senior)
y.hat <- exposure2 * exp (X %*% colMeans(post$beta))
y.rep <- rpois (n, y.hat)

print (mean (y==0))
print (mean (y.rep==0))

## Comparing the data to 1000 replicated datasets

n.sims <- 1000
y.rep <- array (NA, c(n.sims, n))
for (s in 1:n.sims){
  y.hat <- exposure2 * exp (X %*% post$beta[s,])
  y.rep[s,] <- rpois (n, y.hat)
}

 # test statistic 

Test <- function (y){
  mean (y==0)
}
test.rep <- rep (NA, n.sims)
for (s in 1:n.sims){
  test.rep[s] <- Test (y.rep[s,])
}

# p-value
print (mean (test.rep > Test(y)))

## Checking the overdispersed model
# NOTE: can't find same exact data set as ARM book uses..
roaches_overdispersion.sf1 <- stan(file='roaches_overdispersion.stan',
                                   data=dataList.1,
                                   iter=1000, chains=4)
print(roaches_overdispersion.sf1)
post <- extract(roaches_overdispersion.sf1)

 # 1000 replicated datasets 

y.rep <- array (NA, c(n.sims, n))
overdisp <- summary(glm.2)$dispersion
for (s in 1:n.sims){
  y.hat <- exposure2 * exp (X %*% sim.2$coef[s,])
  a <- y.hat/(overdisp-1)              # using R's parametrization for the
  y.rep[s,] <- rnegbin (n, y.hat, a)   # negative binomial distribution
}

test.rep <- rep (NA, n.sims)
for (s in 1:n.sims){
  test.rep[s] <- Test (y.rep[s,])
}

# p-value
print (mean (test.rep > Test(y)))
