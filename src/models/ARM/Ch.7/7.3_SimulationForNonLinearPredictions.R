library(rstan)
library(ggplot2)
congress <- vector ("list", 49)
for (i in 1:49){
  year <- 1896 + 2*(i-1)
  file <- paste ("cong3/", year, ".asc", sep="")
  data.year <- matrix (scan (file), byrow=TRUE, ncol=5)
  data.year <- cbind (rep(year, nrow(data.year)), data.year)
  congress[[i]] <- data.year
}

# Note: download all ".asc" files into your R working directory in a file
# named cong3 for the above command to work

i86 <- (1986-1896)/2 + 1
cong86 <- congress[[i86]]
cong88 <- congress[[i86+1]]
cong90 <- congress[[i86+2]]

v86 <- cong86[,5]/(cong86[,5]+cong86[,6])
bad86 <- cong86[,5]==-9 | cong86[,6]==-9
v86[bad86] <- NA
contested86 <- v86>.1 & v86<.9
inc86 <- cong86[,4]

v88 <- cong88[,5]/(cong88[,5]+cong88[,6])
bad88 <- cong88[,5]==-9 | cong88[,6]==-9
v88[bad88] <- NA
contested88 <- v88>.1 & v88<.9
inc88 <- cong88[,4]

v90 <- cong90[,5]/(cong90[,5]+cong90[,6])
bad90 <- cong90[,5]==-9 | cong90[,6]==-9
v90[bad90] <- NA
contested90 <- v90>.1 & v90<.9
inc90 <- cong90[,4]

jitt <- function (x,delta) {x + runif(length(x), -delta, delta)}

## Plot Figure 7.3
v88.hist <- ifelse (v88<.1, .0001, ifelse (v88>.9, .9999, v88))

frame1 = data.frame(x1=v88.hist)
p1 <- ggplot(frame1,aes(x=x1))  +
      scale_x_continuous("Democratic Share of the Two-Party Vote") +
      labs(title="Congressional Elections in 1988") +
      geom_histogram(colour = "black", fill = "white",binwidth=0.05) +
      theme_bw()
print(p1)
 
## Fitting the model (congress.stan)
## lm (vote.88 ~ vote.86 + incumbency.88)
v86.adjusted <- ifelse (v86<.1, .25, ifelse (v86>.9, .75, v86))
vote.86 <- v86.adjusted[contested88]
incumbency.88 <- inc88[contested88]
vote.88 <- v88[contested88]
ok <- !is.na (vote.86+incumbency.88+vote.88)

dataList.1 <- list(N=length(vote.88[ok]), vote_88=vote.88[ok], vote_86=vote.86[ok],incumbency_88=incumbency.88[ok])
congress.sf1 <- stan(file='congress.stan', data=dataList.1,
                     iter=1000, chains=4)
print(congress.sf1)

fit88.post <- extract(congress.sf1)

## Figure 7.4

# 7.4 (a)
j.v86 <- ifelse (contested86, v86, jitt (v86, .02))
j.v88 <- ifelse (contested88, v88, jitt (v88, .02))

frame1 = data.frame(x1=j.v86[inc88==0],y1=j.v88[inc88==0])
frame2 = data.frame(x2=j.v86[inc88==1],y2=j.v88[inc88==1])
frame3 = data.frame(x3=j.v86[inc88==-1],y3=j.v88[inc88==-1])

dev.new()
p2 <- ggplot() +
      geom_point(data=frame1,aes(x=x1,y=y1),shape=1) +
      geom_point(data=frame2,aes(x=x2,y=y2),shape=16) +
      geom_point(data=frame3,aes(x=x3,y=y3),shape=4) +
      scale_y_continuous("Democratic Vote Share in 1988") +
      scale_x_continuous("Democratic Vote Share in 1986") +
      theme_bw() +
      geom_abline(yintercept=0,slope=1) +
      labs(title="Raw Data (jittered at 0 and 1)")
print(p2)

# 7.4 (b)
v86.adjusted <- ifelse (v86<.1, .25, ifelse (v86>.9, .75, v86))
vote.86 <- v86.adjusted[contested88]
vote.88 <- v88[contested88]
incumbency.88 <- inc88[contested88]

frame4 = data.frame(x1=vote.86[incumbency.88==0],y1=vote.88[incumbency.88==0])
frame5 = data.frame(x2=vote.86[incumbency.88==1],y2=vote.88[incumbency.88==1])
frame6 = data.frame(x3=vote.86[incumbency.88==-1],y3=vote.88[incumbency.88==-1])

dev.new()
p3 <- ggplot() +
      geom_point(data=frame4,aes(x=x1,y=y1),shape=1) +
      geom_point(data=frame5,aes(x=x2,y=y2),shape=16) +
      geom_point(data=frame6,aes(x=x3,y=y3),shape=4) +
      scale_y_continuous("Democratic Vote Share in 1988") +
      scale_x_continuous("Democratic Vote Share in 1986") +
      theme_bw() +
      geom_abline(yintercept=0,slope=1) +
      labs(title="Adjusted data (imputing 0's and 1's to .75)")
print(p3)

## Simulation for inferences and predictions of new data points

incumbency.90 <- inc90
vote.88 <- v88
n.tilde <- length (vote.88)
X.tilde <- cbind (rep (1, n.tilde), vote.88, incumbency.90)

n.sims <- 4000
sim.88 <- fit88.post$beta
y.tilde <- array (NA, c(n.sims, n.tilde))
for (s in 1:n.sims){
  pred <- X.tilde %*% fit88.post$beta[s,]
  ok <- !is.na(pred)
  y.tilde[s,ok] <- rnorm (sum(ok), pred[ok], fit88.post$sigma[s])
}

## Predictive simulation for a nonlinear function of new data

y.tilde.new <- ifelse (y.tilde=="NaN", 0, y.tilde)

dems.tilde <- rowSums (y.tilde.new > .5)
 # or
dems.tilde <- rep (NA, n.sims)
for (s in 1:n.sims){
  dems.tilde[s] <- sum (y.tilde.new[s,] > .5)
}

## Implementation using functions

Pred.88 <- function (X.pred){
  pred <- X.tilde %*% t(fit88.post$beta)
  ok <- !is.na(pred)
  n.pred <- length (pred)
  y.pred <- rep (NA, n.pred)
  y.pred[ok] <- rnorm (sum(ok), pred[ok], fit88.post$sigma)
  return (y.pred)
}

y.tilde <- replicate (1000, Pred.88 (X.tilde))
dems.tilde <- replicate (1000, Pred.88 (X.tilde) > .5)
