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

## Fit the model

dataList.1 <- list(N=n,J=85,y=y,u=u,x=x,county=county)
radon_vary_intercept_a.sf1 <- stan(file='radon_vary_intercept_a.stan',
                                   data=dataList.1, iter=1000, chains=4)
print(radon_vary_intercept_a.sf1,pars = c("a","b","sigma_y", "lp__"))
post <- extract(radon_vary_intercept_a.sf1)
e.a <- colMeans(post$e_a)
omega <- (sd(e.a)/mean(post$sigma_a))^2
omega <- pmin (omega, 1)


## Summary pooling factor for each batch of parameters

dataList.1 <- list(N=n,J=85,y=y,u=u,x=x,county=county)
radon_vary_intercept_b.sf1 <- stan(file='radon_vary_intercept_b.stan',
                                   data=dataList.1, iter=1000, chains=4)
print(radon_vary_intercept_b.sf1,pars = c("a","b","sigma_y", "lp__"))
post <- extract(radon_vary_intercept_b.sf1)

e.y <- (post$e_y)
e.a <- (post$e_a)

lambda.y <- 1 - var (apply (e.y, 2, mean))/ mean (apply (e.y, 1, var))
lambda.a <- 1 - var (apply (e.a, 2, mean))/ mean (apply (e.a, 1, var))

# if slope varies
lambda.b <- 1 - var (apply (e.b, 2, mean))/ mean (apply (e.b, 1, var))
