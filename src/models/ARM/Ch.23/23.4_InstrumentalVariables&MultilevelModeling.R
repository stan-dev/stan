## Read the data & redefine variables
# Data are at http://www.stat.columbia.edu/~gelman/arm/examples/sesame
library(rstan)
library(foreign)
sesame <- read.dta("sesame.dta")

y=sesame$postlet
d=sesame$regular
yt=cbind(sesame$postlet,sesame$regular)
z=sesame$encour
n=nrow(sesame)

siteset=numeric(nrow(sesame))
for(j in 1:2){
for(i in 1:5){
siteset[sesame$site==i & sesame$setting==j]=i+5*(j-1)
}
}
J=9

## Fit the model for example 1

dataList.1 <- list(N=n,J=J,z=z,yt=yt,siteset=siteset)
sesame_street1.sf1 <- stan(file='sesame_street1.stan', data=dataList.1,
                           iter=1000, chains=4)
print(sesame_street1.sf1, pars = c("a","g","b","d","lp__"))


## Fit the model conditioning on pre-treatment variables
## FIXME: MISSING DATA PRETEST VARIABLE

##dataList.2 <- list(N=n,J=J,z=z,yt=yt,siteset=siteset,pretest=pretest)
##sesame_street2.sf1 <- stan(file='sesame_street2.stan', data=dataList.2,
##                           iter=1000, chains=4)
##print(sesame_street2.sf1, pars = c("a","g","b","d","phi_y","phi_t","lp__"))
