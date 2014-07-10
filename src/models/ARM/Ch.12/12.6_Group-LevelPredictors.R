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

## Varying-intercept model w/ group-level predictors
u.full <- u[county]
dataList.3 <- list(N=length(y), y=y,x=x,county=county,u=u.full)
radon_group.sf1 <- stan(file='radon_group.stan', data=dataList.3,
                        iter=1000, chains=4)
print(radon_group.sf1, pars = c("b","beta", "sigma", "lp__"))
post1 <- extract(radon_group.sf1)
post1.ranef <- colMeans(post1$const_coef)
mean1.ranef <- mean(post1.ranef)
post1.beta <- colMeans(post1$beta)
post1.fixef1 <- mean(post1.ranef)

## Plots on Figure 12.5
dataList.4 <- list(N=length(y), y=y,x=x,county=county)
radon_no_pool.sf1 <- stan(file='radon_no_pool.stan', data=dataList.4,
                          iter=1000, chains=4)
print(radon_no_pool.sf1)
post2 <- extract(radon_no_pool.sf1)
post2.ranef <- colMeans(post2$factor)
mean2.ranef <- mean(post2.ranef)
post2.fixef1 <- colMeans(post2$beta)
post2.fixef2 <- mean(post2.ranef)

a.hat.M1 <- post2.fixef2 + post2.ranef - mean2.ranef
b.hat.M1 <- post2.fixef1

a.hat.M2 <- post1.fixef1 + post1.beta[2] * u + post1.ranef - mean1.ranef
b.hat.M2 <- post1.beta[1]

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
radon8.data$m1.int <- a.hat.M1$value[radon8.data$county]
radon8.data$m1.slope <- rep(b.hat.M1$value,209)
radon8.data$m2.int <- a.hat.M2$value[radon8.data$county]
radon8.data$m2.slope <- rep(b.hat.M2$value,209)

p1 <- ggplot(radon8.data, aes(x.jitter, y)) +
     geom_jitter(position = position_jitter(width = .05, height = 0)) +
     scale_x_continuous(breaks=c(0,1), labels=c("0", "1")) +
     geom_abline(aes(intercept = m1.int, slope = m1.slope), size = 0.25,colour="grey10") +
     geom_abline(aes(intercept = m2.int,slope=m2.slope), size = 0.25) +
     facet_wrap(~ county.name, ncol = 4)
print(p1)

# Plot of ests & se's vs. county uranium (Figure 12.6)
a.se.M2 <- se.coef(M2)$county
a.se.M2 <- melt(a.se.M2)
dev.new()

frame1 = data.frame(x1=u,y1=a.hat.M2$value)
limits <- aes(ymax=a.hat.M2$value+a.se.M2$value,ymin=a.hat.M2$value-a.se.M2$value)
p2 <- ggplot(frame1,aes(x=x1,y=y1)) +
      geom_point() +
      theme_bw() +
      geom_pointrange(limits) +
      geom_abline(aes(intercept=fixef(M2)["(Intercept)"],slope=fixef(M2)["u.full"]))
print(p2)
