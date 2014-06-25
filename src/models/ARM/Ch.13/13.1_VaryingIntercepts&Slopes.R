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

## Varying intercept & slopes w/ no group level predictors
## lmer (y ~ x + (1 + x | county))

dataList.3 <- list(N=length(y), y=y,x=x,county=county)
radon_vary_si.sf1 <- stan(file='radon_vary_si.stan', data=dataList.3,
                          iter=1000, chains=4)
print(radon_vary_si.sf1)
post1 <- extract(radon_vary_si.sf1)
post1.ranef <- colMeans(post1$a1)
mean.ranef1 <- mean(post1.ranef)
post1.beta <- colMeans(post1$a2)
post1.ranef2 <- mean(post1.beta)
mean.ranef2 <- mean(post1.ranef2)

 # plots on Figure 13.1
a.hat.M3 <- mean.ranef1 + post1.ranef
b.hat.M3 <- mean.ranef2 + post1.ranef2


b.hat.unpooled.varying <- array (NA, c(J,2))
for (j in 1:J){
  dataList.3 <- list(N=length(y[county==j]), y=y[county==j],x=x[county==j])
  radon_no_pool.sf1 <- stan(file='y_x.stan', data=dataList.3,
                            iter=1000, chains=4)
  post <- extract(radon_no_pool.sf1)
  b.hat.unpooled.varying[j,] <- colMeans(post$beta)
}

dataList.3 <- list(N=length(y), y=y,x=x)
radon_complete_pool.sf1 <- stan(file='y_x.stan', data=dataList.3,
                                iter=1000, chains=4)
post <- extract(radon_complete_pool.sf1)
pool.beta <- colMeans(post$beta)

x.jitter <- x + runif(n,-.05,.05)
display8 <- c (36, 1, 35, 21, 14, 71, 61, 70)  # counties to be displayed
y.range <- range (y[!is.na(match(county,display8))])

radon.data <- data.frame(y, x.jitter, county)
radon8.data <- subset(radon.data, county %in% display8)
radon8.data$county.name <- radon8.data$county
radon8.data$county.name <- factor(radon8.data$county.name,levels=c("36","1","35","21","14","71","61","70"),
                                  labels=c("LAC QUI PARLE", "AITKIN", "KOOCHICHING",
                                      "DOUGLAS", "CLAY", "STEARNS", "RAMSEY",
                                      "ST LOUIS"))
radon8.data$m1.int <- pool.beta[1]
radon8.data$m1.slope <- pool.beta[2]
radon8.data$m2.int <- b.hat.unpooled.varying[,1]
radon8.data$m2.slope <- b.hat.unpooled.varying[,2]
radon8.data$m3.int <- a.hat.M3
radon8.data$m3.slope <- b.hat.M3

p1 <- ggplot(radon8.data, aes(x.jitter, y)) +
     geom_jitter(position = position_jitter(width = .05, height = 0)) +
     scale_x_continuous(breaks=c(0,1), labels=c("0", "1")) +
     geom_abline(aes(intercept = m1.int, slope = m1.slope), size = 0.25,colour="grey10") +
     geom_abline(aes(intercept = m2.int, slope = m2.slope), size = 0.25,colour="grey10") +
     geom_abline(aes(intercept = m3.int,slope=m3.slope), size = 0.25) +
     facet_wrap(~ county.name, ncol = 4)
print(p1)

## Including group level predictors

dataList.4 <- list(N=length(y), y=y,x=x,county=county,u=u)
radon_inter_vary.sf1 <- stan(file='radon_inter_vary.stan', data=dataList.4,
                             iter=1000, chains=4)
print(radon_inter_vary.sf1)
post <- extract(radon_inter_vary.sf1)
ranef1 <- colMeans(post$const_coef)
ranef2 <- colMeans(post$x_coef)
fixef1 <- mean(ranef1)
fixef2 <- mean(ranef2)
sd.ranef1 <- sd(ranef1)
sd.ranef2 <- sd(ranef2)
fixef.beta <- colMeans(post$beta)

a.hat.M4 <- fixef.beta[1] * u + ranef1
b.hat.M4 <- fixef.beta[2] * u + ranef2 
a.se.M4 <- sd.ranef1
b.se.M4 <- sd.ranef2

 # plot on Figure 13.2(a)
lower <- a.hat.M4 - a.se.M4
upper <- a.hat.M4 + a.se.M4

dev.new()
frame2 = data.frame(x1=u,y1=a.hat.M4)
limits <- aes(ymax=upper,ymin=lower)
p2 <- ggplot(frame2,aes(x=x1,y=y1)) +
      geom_point() +
      scale_y_continuous("Regression Intercept") +
      scale_x_log10("County-Level Uranium Measure") +
      theme_bw() +
      geom_abline(intercept=fixef1, slope=fixef.beta[1]) +
      geom_pointrange(limits) +
      labs(title="Intercepts")
print(p2)

 # plot on Figure 13.2(b)
lower <- b.hat.M4 - b.se.M4
upper <- b.hat.M4 + b.se.M4

dev.new()
frame3 = data.frame(x1=u,y1=b.hat.M4)
limits <- aes(ymax=upper,ymin=lower)
p3 <- ggplot(frame3,aes(x=x1,y=y1)) +
      geom_point() +
      scale_y_continuous("Regression Slope") +
      scale_x_log10("County-Level Uranium Measure") +
      theme_bw() +
      geom_abline(intercept=fixef2, slope=fixef.beta[2]) +
      geom_pointrange(limits) +
      labs(title="Slopes")
print(p3)
