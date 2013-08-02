## Read the data
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/radon

# The R codes & data files should be saved in the same directory for
# the source command to work

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

if (!exists("anova_radon_nopred.sm")) {
    if (file.exists("anova_radon_nopred.sm.RData")) {
        load("anova_radon_nopred.sm.RData", verbose = TRUE)
    } else {
        rt <- stanc("anova_radon_nopred.stan", model_name = "anova_radon_nopred")
        anova_radon_nopred.sm <- stan_model(stanc_ret = rt)
        save(anova_radon_nopred.sm, file = "anova_radon_nopred.sm.RData")
    }
}

dataList.1 <- list(N=length(y), y=y, county=county,J=85)
anova_radon_nopred.sf1 <- sampling(anova_radon_nopred.sm, dataList.1)
print(anova_radon_nopred.sf1,pars = c("a","sigma_y","lp__"))
