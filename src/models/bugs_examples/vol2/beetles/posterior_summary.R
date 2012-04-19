## transform alpha.star back to alpha
## note that three models can be fitted, i.e., logit, probit and cloglog

library(coda) 
logit <- function(x) log(x / (1 - x)); 
mean_x <- mean(c(1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839))
psamp <- read.csv(file = "samples.csv", header = TRUE, comment.char = '#'); 
alpha <- psamp["alpha"]; beta <- psamp["beta"]
rhat <- psamp[paste("rhat", 1:8, sep=".")]
poi <- cbind(alpha, beta, rhat)

poi <- as.mcmc(poi)
summary(poi) 


 
