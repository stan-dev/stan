## take a look at the samples and compare with results computed 
## in other program. 

library(coda) 
post <- read.csv(file = "samples.csv", header = TRUE, comment.char = '#'); 

poi <- post[, c("alpha", "beta1", "beta2", "sigma")] 
poi <- as.mcmc(poi)
summary(poi) 

# copied from jags example 
"benchstats" <-
structure(c(0.578583954000001, -0.0461005503250001, 0.00702949669105,
0.1230820353658, 0.0654185681985203, 0.0149358418576360, 0.00305161514347347,
0.0859365730618193, 0.00146280365482597, 0.000333975576948609,
6.82361890197445e-05, 0.00192160019119605, 0.00161378145150198,
0.000363947519384889, 7.65060538223112e-05, 0.00267000525994563
), .Dim = as.integer(c(4, 4)), .Dimnames = list(c("alpha", "beta1",
"beta2", "sigma"), c("Mean", "SD", "Naive SE", "Time-series SE"
)))

print(benchstats) 
