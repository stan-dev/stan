## Read the data & redefine variables
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/dogs

library(rstan)
library(ggplot2)
library(reshape2)

y1 <- as.matrix (read.table ("dogs.dat"), nrows=30, ncol=25)
y <- ifelse (y1[,]=="S",1,0)
n.dogs <- nrow(y)
n.trials <- ncol(y)

## Calling predictive replications in Bugs

dataList.1 <- list(n_dogs=n.dogs,n_trials=n.trials,y=y)
dogs.sf1 <- stan(file='dogs.stan', data=dataList.1, iter=1000, chains=4)
print(dogs.sf1, pars = c("beta","lp__"))
post <- extract(dogs.sf1)
beta <- colMeans(post$beta)

## Pedictive replications in R
n.sims <- 4000
y.rep <- array (NA, c(n.sims, n.dogs, n.trials))
for (j in 1:n.dogs){
  n.avoid.rep <- rep (0, n.sims)
  n.shock.rep <- rep (0, n.sims)
  for (t in 1:n.trials){  
    p.rep <- invlogit (beta[1] + beta[2]*n.avoid.rep + beta[3]*n.schok.rep)
    y.rep[,j,t] <- rbinom (n.sims, 1, p.rep)
    n.avoid.rep <- n.avoid.rep + 1 - y.rep[,j,t] 
    n.shock.rep <- n.shock.rep + y.rep[,j,t] 
    }
}

## Direct comparison of simulated to real data
dogsort <- function (y){
  n.dogs <- nrow(y)
  n.trials <- ncol(y)
  last.shock <- rep (NA, n.dogs)
  for (j in 1:n.dogs){
    last.shock[j] <- max ((1:n.trials)[y[j,]==1])
  }
  y[order(last.shock),]
}


## More focused model checks

 # Figure 24.2a

test <- function (data){
  colMeans (1-data)
}
y.sim <- melt(y.rep)
frame = data.frame(x1=0:(n.trials-1),y1=test(y.sim[,,],newpid=newpid))
p1 <- ggplot(frame,aes(x=x1,y=y1)) +
    geom_point() +
    scale_y_continuous("sqrt (CD4%)") +
    scale_x_continuous("Time (years)") +
    theme_bw() +
    labs(title="Observed Data")
for (j in 1:84) {
  p1 <- p1 + geom_line(data=frame[frame$newpid==unique.pid[j],])
}
print(p1)

  
 # Figure 24.2b
test.diff <- function (data, data.rep){
  test (data) - test (data.rep)
}

diff.range <- NULL
for (s in 1:20){
  diff.range <- range (diff.range, test.diff (y, y.rep[s,,]))
}

dev.new()
y.sim <- melt(y.rep)
frame = data.frame(x1=0:(n.trials-1),y1=test.diff(y,y.sim[,,]),newpid=newpid)
p2 <- ggplot(frame,aes(x=x1,y=y1)) +
    geom_point() +
    scale_y_continuous("sqrt (CD4%)") +
    scale_x_continuous("Time (years)") +
    theme_bw() +
    labs(title="Observed Data")
for (j in 1:84) {
  p2 <- p2 + geom_line(data=frame[frame$newpid==unique.pid[j],])
}
print(p2)
