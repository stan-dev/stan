## take a look at the samples and compare with results computed 
## in other program. 

library(coda) 

N <- 10; 
post <- read.csv(file = "samples.csv", header = TRUE, comment.char = '#')  

mystep <- function(x) ifelse(x >= 0, 1, 0)  
theta <- exp(post[, "phi"])
equiv <- mystep(theta - 0.8) - mystep(theta - 1.2)

summary(as.mcmc(post[, "phi"])) 

poi <- cbind(equiv, post[, c("mu", "phi", "pi", "sigma1", "sigma2")], theta)  
pars <- c("equiv", "mu", "phi", "pi", "sigma1", "sigma2", "theta") 
colnames(poi) <- pars 
summary(as.mcmc(poi)) 
plot(as.mcmc(poi)) 


## run jags 

library(BUGSExamples)

ex <- list(name = "Equiv", parameters = pars, 
           nSample = 10000, nBurnin = 1000, nThin = 1, 
           nChain = 3)

jagspost <- runExample(ex, engine = 'JAGS') 
summary(jagspost$coda) 
plot(jagspost$coda);
