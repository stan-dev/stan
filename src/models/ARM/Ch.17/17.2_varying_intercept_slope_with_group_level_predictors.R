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

## Get the county-level predictor
srrs2.fips <- srrs2$stfips*1000 + srrs2$cntyfips
cty <- read.table ("cty.dat", header=T, sep=",")
usa.fips <- 1000*cty[,"stfips"] + cty[,"ctfips"]
usa.rows <- match (unique(srrs2.fips[mn]), usa.fips)
uranium <- cty[usa.rows,"Uppm"]
u <- log (uranium)

## Varying-intercept model w/ group-level predictors
u.full <- u[county]

 # radon varying intercept and slope model
dataList.1 <- list(N=length(y), y=y, county=county, J=J, x=x, u=u.full)
radon_vary_inter_slope.sf1 <- stan(file='17.2_radon_vary_inter_slope.stan', data=dataList.1,
                            iter=1000, chains=4)
print(radon_vary_inter_slope.sf1)

 # radon correlation model
radon_correlation.sf1 <- stan(file='17.2_radon_correlation.stan', data=dataList.1,
                            iter=1000, chains=4)
print(radon_correlation.sf1)

W <- diag (2)
dataList.2 <- list(N=length(y), y=y, county=county, J=J, x=x, W=W, u=u.full)

 # radon Scaled inverse-Wishart model
radon_wishart.sf1 <- stan(file='17.2_radon_wishart.stan', data=dataList.2,
                            iter=1000, chains=4)
print(radon_wishart.sf1)
