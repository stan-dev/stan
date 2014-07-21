library(rstan)
library(ggplot2)

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

## Complete pooling regression
dataList.1 <- list(N=length(y), y=y,x=x)
radon_complete_pool.sf1 <- stan(file='radon_complete_pool.stan',
                                data=dataList.1,
                                iter=1000, chains=4)
print(radon_complete_pool.sf1)
post.pooled <- extract(radon_complete_pool.sf1)
pooled <- colMeans(post.pooled$beta)

## No pooling regression

dataList.2 <- list(N=length(y), y=y,x=x,county=county)
radon_no_pool.sf1 <- stan(file='radon_no_pool.stan', data=dataList.2,
                          iter=1000, chains=4)
print(radon_no_pool.sf1)
post.unpooled <- extract(radon_no_pool.sf1)
unpooled <- colMeans(post.unpooled$a)
sd.unpooled <- rep(NA,85)
for (n in 1:85) {
  sd.unpooled[n] <- sd(post.unpooled$a[,n]) 
}

## Comparing-complete pooling & no-pooling (Figure 12.2)
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
radon8.data$pooled.int <- pooled[1]
radon8.data$pooled.slope <- pooled[2]
radon8.data$unpooled.int <- unpooled[radon8.data$county]
radon8.data$unpooled.slope <- mean(post.unpooled$beta)

p1 <- ggplot(radon8.data, aes(x.jitter, y)) +
      geom_jitter(position = position_jitter(width = .05, height = 0)) +
      scale_x_continuous(breaks=c(0,1), labels=c("0", "1")) +
      geom_abline(aes(intercept = pooled.int, slope = pooled.slope), linetype = "dashed") +
      geom_abline(aes(intercept = unpooled.int, slope = unpooled.slope), size = 0.25) +
      facet_wrap(~ county.name, ncol = 4)
print(p1)

## No-pooling ests vs. sample size (plot on the left on figure 12.3)
sample.size <- as.vector (table (county))
sample.size.jittered <- sample.size*exp (runif (J, -.1, .1))
dev.new()
frame1 = data.frame(x1=sample.size.jittered,y1=unpooled)
limits <- aes(ymax=unpooled+sd.unpooled, ymin=unpooled-sd.unpooled)
p2 <- ggplot(frame1,aes(x=x1,y=y1)) +
      geom_point() +
      scale_y_continuous("estimated intercept alpha (no pooling)") +
      scale_x_log10("Sample Size in County j") +
      theme_bw() +
      geom_pointrange(limits)
print(p2)
