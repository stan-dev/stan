## transform alpha.star back to alpha
## note that three models can be fitted, i.e., logit, probit and cloglog

library(coda) 
logit <- function(x) log(x / (1 - x)); 
mean_x <- mean(c(1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839))
psamp <- read.csv(file = "samples.csv"); 
alpha.star <- psamp[,1]; beta <- psamp[,2]
alpha <- alpha.star - beta*mean_x
poi <- cbind(alpha, beta)
## assuming the order of variables in samples.csv are the same as model
## specification file 

poi <- as.mcmc(poi)
summary(poi) 


 
