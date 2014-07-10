library(rstan)

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

 # radon varying intercept and slope model
dataList.1 <- list(N=length(y), y=y, county=county, J=J, x=x)
radon_vary_inter_slope.sf1 <- stan(file='17.1_radon_vary_inter_slope.stan', data=dataList.1,
                            iter=1000, chains=4)
print(radon_vary_inter_slope.sf1)

 # radon correlation model
radon_correlation.sf1 <- stan(file='17.1_radon_correlation.stan', data=dataList.1,
                            iter=1000, chains=4)
print(radon_correlation.sf1)

 # radon multiple varying coefficients model
X <- cbind(1,x)
W <- diag(2)
dataList.1 <- list(N=length(y), y=y, county=county, J=J, X=X, K=2, W=W)
radon_multi_varying_coef.sf1 <- stan(file='17.1_radon_multi_varying_coef.stan', data=dataList.1,
                            iter=1000, chains=4)
print(radon_multi_varying_coef.sf1)

W <- diag (2)
dataList.2 <- list(N=length(y), y=y, county=county, J=J, x=x, W=W)

 # radon Scaled inverse-Wishart model
radon_wishart.sf1 <- stan(file='17.1_radon_wishart.stan', data=dataList.2,
                            iter=1000, chains=4)
print(radon_wishart.sf1)

 # radon Scaled inverse-Wishart model 2
radon_wishart2.sf1 <- stan(file='17.1_radon_wishart2.stan', data=dataList.2,
                            iter=1000, chains=4)
print(radon_wishart2.sf1)
