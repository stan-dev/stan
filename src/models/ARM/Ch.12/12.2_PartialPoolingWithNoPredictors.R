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

 # varying-intercept model, no predictors
if (!file.exists("radon_intercept.sm.RData")) {
    rt <- stanc("radon_intercept.stan", model_name="radon_intercept")
    radon_intercept.sm <- stan_model(stanc_ret=rt)
    save(radon_intercept.sm, file="radon_intercept.sm.RData")
} else {
    load("radon_intercept.sm.RData", verbose=TRUE)
}

dataList.1 <- list(N=length(y), y=y, county=county)
radon_intercept.sf1 <- sampling(radon_intercept.sm, dataList.1)
print(radon_intercept.sf1)

post <- extract(radon_intercept.sf1)
mean.a <- rep (NA, 85)
sd.a <- rep (NA, 85)
for (n in 1:85) {
  mean.a[n] <- mean(post$a[,n])
  sd.a[n] <- sd(post$a[,n])
}
## Figure 12.1 (a)
frame1 = data.frame(x1=sample.size.jittered,y1=cty.mns,x2=sample.size.jittered[36],y2=cty.mns[36])
limits <- aes(ymax=cty.mns + cty.sds,ymin=cty.mns - cty.sds)
m2 <- ggplot(frame1,aes(x=x1,y=y1))
m2 + geom_point(aes(x=x2,y=y2),shape=1,size=30) + scale_y_continuous("Avg. Log Radon in County j") + scale_x_log10("Sample Size in County j") + theme_bw() + geom_pointrange(limits) + labs(title="No Pooling")

## Figure 12.1 (b)
frame2 = data.frame(x1=sample.size.jittered,y1=mean.a,x2=sample.size.jittered[36],y2=mean.a[36])
limits <- aes(ymax=mean.a+sd.a, ymin=mean.a-sd.a)
m2 <- ggplot(frame2,aes(x=x1,y=y1))
m2 + geom_point(aes(x=x2,y=y2),shape=1,size=30) + scale_y_continuous("Avg. Log Radon in County j") + scale_x_log10("Sample Size in County j") + theme_bw() + geom_pointrange(limits) + labs(title="Multilevel Model")
