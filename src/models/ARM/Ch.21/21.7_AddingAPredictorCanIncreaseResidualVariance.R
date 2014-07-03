library(rstan)
library(ggplot2)
## Read the data
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/radon

# The R codes & data files should be saved in the same directory for
# the source command to work

srrs2 <- read.table ("srrs2.dat", header=T, sep=",")
mn <- srrs2$state=="MN"
radon <- srrs2$activity[mn]
log.radon <- log (ifelse (radon==0, .1, radon))
floor <- srrs2$floor[mn]       # 0 for basement, 1 for first floor
n <- length(radon)
y <- log.radon
x <- floor

# get county index variable
county.name <- as.vector(srrs2$county[mn])
uniq <- unique(county.name)
J <- length(uniq)
county <- rep (NA, J)
for (i in 1:J){
  county[county.name==uniq[i]] <- i
}

 # no predictors
ybarbar = mean(y)

sample.size <- as.vector (table (county))
sample.size.jittered <- sample.size*exp (runif (J, -.1, .1))
cty.mns = tapply(y,county,mean)
cty.vars = tapply(y,county,var)
cty.sds = mean(sqrt(cty.vars[!is.na(cty.vars)]))/sqrt(sample.size)
cty.sds.sep = sqrt(tapply(y,county,var)/sample.size)

## Get the county-level predictor
srrs2.fips <- srrs2$stfips*1000 + srrs2$cntyfips
cty <- read.table ("cty.dat", header=T, sep=",")
usa.fips <- 1000*cty[,"stfips"] + cty[,"ctfips"]
usa.rows <- match (unique(srrs2.fips[mn]), usa.fips)
uranium <- cty[usa.rows,"Uppm"]
u <- log (uranium)
u.full <- u[county]


# varying-intercept model (NO FLOOR!)

dataList.1 <- list(N=n,J=85,y=y,u=u.full,county=county)
radon_vary_intercept_nofloor.sf1 <- stan(file='radon_vary_intercept_nofloor.stan', data=dataList.1, iter=1000, chains=4)
print(radon_vary_intercept_nofloor.sf1,pars = c("a","b","sigma_y", "lp__"))


# add floor as an individual-level predictor

dataList.1 <- list(N=n,J=85,y=y,u=u.full,x=x,county=county)
radon_vary_intercept_floor.sf1 <- stan(file'radon_vary_intercept_floor.stan',
                                       data=dataList.1, iter=1000, chains=4)
print(radon_vary_intercept_floor.sf1,pars = c("a","b","sigma_y", "lp__"))


# add houses with no basement as group-level predictor

x.mean <- rep (NA, J)
for (j in 1:J){
  x.mean[j] <- mean(x[county==j])
}
x.mean.full <- x.mean[county]

dataList.1 <- list(N=n,J=85,y=y,u=u.full,x=x,county=county,x_mean=x.mean.full)
radon_vary_intercept_floor2.sf1 <- stan(file='radon_vary_intercept_floor2.stan',
                                        data=dataList.1, iter=1000, chains=4)
print(radon_vary_intercept_floor2.sf1,pars = c("a","b","sigma_y", "lp__"))

# Figure 21.9
b.00 <- -2
a.00 <- 1
rand1 <- rnorm(2,0,1)
rand1 <- (rand1-mean(rand1))/sd(rand1)
rand2 <- rnorm(18,0,1)
rand2 <- (rand2-mean(rand2))/sd(rand2)

frame1 = data.frame(x1=rep(c(1,0),c(10,10)),
                    y1=a.00 + rep(c(1,0),c(2,18))*b.00 + c(rand1,rand2)*0.8)
p1 <- ggplot(frame1,aes(x=x1,y=y1)) +
    geom_jitter(position = position_jitter(width = 0.05, height = 0)) +
    geom_abline(aes(intercept = a.00, slope = b.00)) +
    scale_x_continuous("Floor") +
    scale_y_continuous("Log Radon level") +
    labs(title="Naturally Low-Radon County") +
    theme_bw()
print(p1)

dev.new()
a.00 <- 1.8
rand1 <- rnorm(10,0,1)
rand1 <- (rand1-mean(rand1))/sd(rand1)
rand2 <- rnorm(10,0,1)
rand2 <- (rand2-mean(rand2))/sd(rand2)

frame2 = data.frame(x1=rep(c(1,0),c(10,10)),
                    y1=a.00 + rep(c(1,0),c(10,10))*b.00 + c(rand1,rand2)*0.8)
p2 <- ggplot(frame2,aes(x=x1,y=y1)) +
    geom_jitter(position = position_jitter(width = 0.05, height = 0)) +
    geom_abline(aes(intercept = a.00, slope = b.00)) +
    scale_x_continuous("Floor") +
    scale_y_continuous("Log Radon level") +
    labs(title="Intermediate County") +
    theme_bw()
print(p2)

dev.new()
a.00 <- 2.6
rand1 <- rnorm(18,0,1)
rand1 <- (rand1-mean(rand1))/sd(rand1)
rand2 <- rnorm(2,0,1)
rand2 <- (rand2-mean(rand2))/sd(rand2)

frame3 = data.frame(x1=rep(c(1,0),c(18,2)),
                    y1=a.00 + rep(c(1,0),c(18,2))*b.00 + c(rand1,rand2)*0.8)
p3 <- ggplot(frame3,aes(x=x1,y=y1)) +
    geom_jitter(position = position_jitter(width = 0.05, height = 0)) +
    geom_abline(aes(intercept = a.00, slope = b.00)) +
    scale_x_continuous("Floor") +
    scale_y_continuous("Log Radon level") +
    labs(title="Naturally High-Radon County") +
    theme_bw()
print(p3)
